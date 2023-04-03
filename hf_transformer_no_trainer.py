from pathlib import Path
import random
import math
import logging
import os
import json
from itertools import chain
from tqdm.auto import tqdm
from datasets import load_dataset
import datasets
import transformers
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    GPT2Config,
    default_data_collator,
    get_scheduler,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import git
from torch.utils.data import DataLoader
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from config import Config

logger = get_logger(__name__)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def tokenize_function(examples):
    return tokenizer(examples[text_column_name])


def group_texts(examples):
    """Main data processing function that will concatenate
    all texts from our dataset and generate chunks of block_size.
    Args:
        examples (str): text to process

    Returns:
        str: processed text
    """
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


if __name__ == "__main__":
    # Better way: hydra+config files
    configs = Config()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwconfigs = {}

    if configs.with_tracking:
        accelerator_log_kwconfigs["log_with"] = configs.report_to
        accelerator_log_kwconfigs["logging_dir"] = configs.output_dir

    accelerator = Accelerator(
        gradient_accumulation_steps=configs.gradient_accumulation_steps,
        **accelerator_log_kwconfigs,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if configs.seed is not None:
        set_seed(configs.seed)

    # Handle the repository creation
    if accelerator.is_main_process and configs.output_dir is not None:
        os.makedirs(configs.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Download/Reload the dataset
    if (
        Path(configs.train_dataset_location).is_file()
        and Path(configs.test_dataset_location).is_file()
    ):
        # Reload the dataset saved in local files
        train_set = load_dataset("json", data_files=configs.train_dataset_location)[
            "train"
        ]
        test_set = load_dataset("json", data_files=configs.test_dataset_location)[
            "train"
        ]

        if configs.tokenizer_name:
            tokenizer = AutoTokenizer.from_pretrained(
                configs.tokenizer_name, use_fast=not configs.use_slow_tokenizer
            )
        elif configs.model_name:
            tokenizer = AutoTokenizer.from_pretrained(
                configs.model_name, use_fast=not configs.use_slow_tokenizer
            )
        else:
            raise ValueError(
                "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            )

        tokenizer_length = len(tokenizer)

        # Initialize the model_config
        if configs.model_config_name:
            config = AutoConfig.from_pretrained(configs.model_config_name)
        elif configs.model_name:
            config = AutoConfig.from_pretrained(configs.model_name)
        else:
            raise ValueError("No valid model name or config_name")

    else:
        # Downloading and loading a dataset from the hub.
        if configs.dataset_name is not None:
            raw_datasets = load_dataset(
                configs.dataset_name, configs.dataset_config_name
            )
            if "validation" not in raw_datasets.keys():
                raw_datasets["validation"] = load_dataset(
                    configs.dataset_name,
                    configs.dataset_config_name,
                    split=f"train[:{configs.validation_split_percentage}%]",
                )
                raw_datasets["train"] = load_dataset(
                    configs.dataset_name,
                    configs.dataset_config_name,
                    split=f"train[{configs.validation_split_percentage}%:]",
                )
        else:
            raise ValueError(
                "Need either a dataset name or a training/validation file."
            )

        # Initialize the tokenizer
        if configs.tokenizer_name:
            tokenizer = AutoTokenizer.from_pretrained(
                configs.tokenizer_name, use_fast=not configs.use_slow_tokenizer
            )
        elif configs.model_name:
            tokenizer = AutoTokenizer.from_pretrained(
                configs.model_name, use_fast=not configs.use_slow_tokenizer
            )
        else:
            raise ValueError(
                "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            )

        tokenizer_length = len(tokenizer)

        # Initialize the model_config
        if configs.model_config_name:
            config = AutoConfig.from_pretrained(configs.model_config_name)
        elif configs.model_name:
            config = AutoConfig.from_pretrained(configs.model_name)
        else:
            raise ValueError("No valid model name or config_name")

        # Tokenize the dataset
        column_names = raw_datasets["train"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]

        with accelerator.main_process_first():
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=configs.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not configs.overwrite_cache,
                desc="Running tokenizer on dataset",
            )

        # Define the block size
        if configs.block_size is None:
            block_size = tokenizer.model_max_length
            if block_size > 1024:
                logger.warning(
                    "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                    " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                    " override this default in the config file."
                )
            block_size = 1024
        else:
            if configs.block_size > tokenizer.model_max_length:
                logger.warning(
                    f"The block_size passed ({configs.block_size}) is larger than the maximum length for the model"
                    f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
                )
            block_size = min(configs.block_size, tokenizer.model_max_length)

        # Preprocess the tokenized dataset
        with accelerator.main_process_first():
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=configs.preprocessing_num_workers,
                load_from_cache_file=not configs.overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}",
            )

        # Split and save the datasets
        train_set = lm_datasets["train"].shuffle(seed=configs.seed)
        train_set.to_json(configs.train_dataset_location)
        test_set = lm_datasets["validation"].shuffle(seed=configs.seed)
        test_set.to_json(configs.test_dataset_location)

    # Load a HuggingFace pretrained causal language model
    # configuration = RobertaConfig()
    # model = RobertaModel(configuration)
    # model_config = GPT2Config()
    # model = AutoModelForCausalLM.from_config(model_config)

    if configs.model_name:
        # Always from scratch for the moment (no need pretrained for now)
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if tokenizer_length > embedding_size:
        model.resize_token_embeddings(tokenizer_length)

    # Name of the run
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    configs.run_name = f"{configs.run_name}_{sha}"

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_set)), 3):
        logger.info(f"Sample {index} of the training set: {train_set[index]}.")

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_set,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=configs.per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        test_set,
        collate_fn=default_data_collator,
        batch_size=configs.per_device_eval_batch_size,
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": configs.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, lr=configs.learning_rate
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / configs.gradient_accumulation_steps
    )
    if configs.max_train_steps is None:
        configs.max_train_steps = configs.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=configs.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=configs.num_warmup_steps * configs.gradient_accumulation_steps,
        num_training_steps=configs.max_train_steps
        * configs.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    (
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / configs.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        MAX_TRAIN_STEPS = configs.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    NUM_TRAIN_EPOCHS = math.ceil(MAX_TRAIN_STEPS / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = configs.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if configs.with_tracking:
        experiment_config = vars(configs)
        # TensorBoard cannot log Enums, need the raw value
        # experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers(
            configs.project_name,
            experiment_config,
            init_kwargs={"wandb": {"name": configs.run_name}},
        )

    # Train!
    total_batch_size = (
        configs.per_device_train_batch_size
        * accelerator.num_processes
        * configs.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_set)}")
    logger.info(f"  Num Epochs = {configs.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {configs.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {configs.gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {configs.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(configs.max_train_steps), disable=not accelerator.is_local_main_process
    )
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if configs.resume_from_checkpoint:
        if (
            configs.resume_from_checkpoint is not None
            or configs.resume_from_checkpoint != ""
        ):
            accelerator.print(
                f"Resumed from checkpoint: {configs.resume_from_checkpoint}"
            )
            accelerator.load_state(configs.resume_from_checkpoint)
            path = os.path.basename(configs.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[
                -1
            ]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = (
                int(training_difference.replace("step_", ""))
                * configs.gradient_accumulation_steps
            )
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(starting_epoch * num_update_steps_per_epoch)
    completed_steps = starting_epoch * num_update_steps_per_epoch

    for epoch in range(starting_epoch, configs.num_train_epochs):
        print(f"Epoch: {epoch}")
        model.train()
        if configs.with_tracking:
            total_loss = 0
        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            if configs.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    if step % configs.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                        completed_steps += 1
                    continue

            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                training_perplexity = math.exp(loss)
                # We keep track of the loss at each epoch
                if configs.with_tracking:
                    total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps }"
                    if configs.output_dir is not None:
                        output_dir = os.path.join(configs.output_dir, output_dir)
                    accelerator.save_state(output_dir)
            if completed_steps >= MAX_TRAIN_STEPS:
                break

            if (step - 1) % 100 == 0 and configs.with_tracking:
                print((step - 1) % 100)
                train_loss = total_loss.item() / step
                logger.info(
                    f"epoch {epoch} step {step}: perplexity: {training_perplexity} training_loss: {train_loss}"
                )
                accelerator.log(
                    {
                        "training_perplexity": training_perplexity,
                        "train_loss": train_loss,  # len(train_dataloader),
                        "epoch": epoch,
                        "step": completed_steps,
                    },
                    step=completed_steps,
                )
                # Add save batch and weights model here (overwrite if no spike)
            if configs.checkpointing_steps == "step":
                output_dir = f"epoch_{epoch}"
                if configs.output_dir is not None:
                    output_dir = os.path.join(configs.output_dir, output_dir)
                accelerator.save_state(output_dir)

        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses.append(
                accelerator.gather_for_metrics(
                    loss.repeat(configs.per_device_eval_batch_size)
                )
            )

        losses = torch.cat(losses)
        try:
            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")

        logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")

        if configs.with_tracking:
            accelerator.log(
                {
                    "perplexity": perplexity,
                    "eval_loss": eval_loss,
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )

        if configs.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if configs.output_dir is not None:
                output_dir = os.path.join(configs.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if configs.with_tracking:
        accelerator.end_training()

    if configs.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            configs.output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(configs.output_dir)
            with open(os.path.join(configs.output_dir, "all_results.json"), "w") as f:
                json.dump({"perplexity": perplexity}, f)
