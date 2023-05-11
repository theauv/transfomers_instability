from itertools import chain
import logging
import math
import os
from pathlib import Path
import shutil
from typing import Dict
from pynvml import *
from tqdm.auto import tqdm
import yaml
import git
import torch
from accelerate import Accelerator, DistributedType
from accelerate.utils import set_seed
from datasets import load_dataset
import datasets
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    default_data_collator,
    get_scheduler,
)
from torch.utils.data import DataLoader
from .config_dataclass import Config


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()


def convert_yaml_configs(overrides: Dict = {}, configs_directory_path="configs"):
    configs = {}
    for filename in os.listdir(configs_directory_path):
        f = os.path.join(configs_directory_path, filename)
        with open(f, "r") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        configs.update(config)
    configs.update(overrides)

    config_dataclass = Config(**configs)

    return config_dataclass


def load_dataset_for_training(
    data_location,
    train_data_location,
    test_data_location,
    debug_set,
    train_debug_set_location,
    test_debug_set_location,
    tokenizer_name,
    model_name,
    use_slow_tokenizer,
    model_config_name,
    dataset_name,
    dataset_config_name,
    validation_split_percentage,
    preprocessing_num_workers,
    overwrite_cache,
    logger,
    block_size,
    seed,
):
    def tokenize_function(examples):
        return tokenizer(examples[text_column_name], max_length=1024, truncation=True)

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

    # Download/Reload the dataset
    data_location = os.path.join(data_location, Path(f"data_{dataset_name}_{dataset_config_name}"))

    if debug_set:
        train_dataset_location = os.path.join(data_location, train_debug_set_location)
        test_dataset_location = os.path.join(data_location, test_debug_set_location)
    else:
        train_dataset_location = os.path.join(data_location, train_data_location)
        test_dataset_location = os.path.join(data_location, test_data_location)

    if Path(train_dataset_location).is_file() and Path(test_dataset_location).is_file() and False:
        # Reload the dataset saved in local files

        print("#####LOADING FROM SAVED DATASET#####")

        train_set = load_dataset("json", data_files=train_dataset_location)["train"]
        test_set = load_dataset("json", data_files=test_dataset_location)["train"]

        if tokenizer_name:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name, use_fast=not use_slow_tokenizer
            )
        elif model_name:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, use_fast=not use_slow_tokenizer
            )
        else:
            raise ValueError(
                "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            )

        tokenizer_length = len(tokenizer)

        # Initialize the model_config
        if model_config_name:
            model_config = AutoConfig.from_pretrained(model_config_name)
        elif model_name:
            model_config = AutoConfig.from_pretrained(model_name)
        else:
            raise ValueError("No valid model name or config_name")

    else:
        print("#####DOWNLOADING FROM THE HUB#####")
        # Downloading and loading a dataset from the hub.
        if dataset_name == "openwebtext":
            dataset = load_dataset(dataset_name)
            # owt by default only contains the 'train' split, so create a test split
            raw_datasets = dataset["train"].train_test_split(test_size=validation_split_percentage/100, seed=seed, shuffle=True)
            raw_datasets["validation"] = raw_datasets.pop('test') # rename the test split to val

        elif dataset_name is not None:
            if debug_set:
                raw_datasets = load_dataset(dataset_name, dataset_config_name)
                raw_datasets["validation"] = load_dataset(
                    dataset_name,
                    dataset_config_name,
                    split="train[:100]",
                )
                raw_datasets["train"] = load_dataset(
                    dataset_name,
                    dataset_config_name,
                    split="train[100:1000]",
                )
            else:
                raw_datasets = load_dataset(dataset_name, dataset_config_name)
                if "validation" not in raw_datasets.keys():
                    raw_datasets["validation"] = load_dataset(
                        dataset_name,
                        dataset_config_name,
                        split=f"train[:{validation_split_percentage}%]",
                    )
                    raw_datasets["train"] = load_dataset(
                        dataset_name,
                        dataset_config_name,
                        split=f"train[{validation_split_percentage}%:]",
                    )
        else:
            raise ValueError(
                "Need either a dataset name or a training/validation file."
            )

        # Initialize the tokenizer
        if tokenizer_name:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name, use_fast=not use_slow_tokenizer
            )
        elif model_name:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, use_fast=not use_slow_tokenizer
            )
        else:
            raise ValueError(
                "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            )

        tokenizer_length = len(tokenizer)

        # Initialize the model_config
        if model_config_name:
            model_config = AutoConfig.from_pretrained(model_config_name)
        elif model_name:
            model_config = AutoConfig.from_pretrained(model_name)
        else:
            raise ValueError("No valid model name or config_name")

        # Tokenize the dataset
        column_names = raw_datasets["train"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]

        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not overwrite_cache,
            desc="Running tokenizer on dataset",
        )

        # Define the block size
        if block_size is None:
            block_size = tokenizer.model_max_length
            if block_size > 1024:
                logger.warning(
                    "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                    " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                    " override this default in the config file."
                )
            block_size = 1024
        else:
            if block_size > tokenizer.model_max_length:
                logger.warning(
                    f"The block_size passed ({block_size}) is larger than the maximum length for the model"
                    f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
                )
            block_size = min(block_size, tokenizer.model_max_length)

        # Preprocess the tokenized dataset
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=preprocessing_num_workers,
            load_from_cache_file=not overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )

        # # Split and save the datasets
        # if not Path(data_location).is_dir():
        #     os.makedirs(f"{data_location}/train")
        #     os.makedirs(f"{data_location}/valid")

        train_set = lm_datasets["train"].shuffle(seed=seed)
        # train_set.to_json(train_dataset_location)
        test_set = lm_datasets["validation"].shuffle(seed=seed)
        # test_set.to_json(test_dataset_location)

    return train_set, test_set, tokenizer_length, model_config


def train(
    logger,
    accelerator: Accelerator,
    train_set,
    train_dataloader,
    eval_dataloader,
    model,
    optimizer,
    lr_scheduler,
    num_update_steps_per_epoch,
    per_device_train_batch_size,
    gradient_accumulation_steps,
    num_train_epochs,
    max_train_steps,
    resume_from_checkpoint,
    with_tracking,
    spike_threshold,
    checkpointing_steps,
    config_output_dir,
    per_device_eval_batch_size,
):
    # Train!
    total_batch_size = (
        per_device_train_batch_size
        * accelerator.num_processes
        * gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_set)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(max_train_steps), disable=not accelerator.is_local_main_process
    )
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if resume_from_checkpoint:
        if resume_from_checkpoint is not None or resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {resume_from_checkpoint}")
            accelerator.load_state(resume_from_checkpoint)
            path = os.path.basename(resume_from_checkpoint)
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
                * gradient_accumulation_steps
            )
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(starting_epoch * num_update_steps_per_epoch)
    completed_steps = starting_epoch * num_update_steps_per_epoch

    for epoch in range(starting_epoch, num_train_epochs):
        print(f"Epoch: {epoch}")
        model.train()
        if with_tracking:
            total_loss = 0
        batches = {}
        spike_detected = False
        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            if resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    if step % gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                        completed_steps += 1
                    continue

            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                # training_perplexity = math.exp(loss) or here ??
                # We keep track of the loss at each epoch
                if with_tracking:
                    total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            # Not supported for now
            # if isinstance(checkpointing_steps, int):
            #     if completed_steps % checkpointing_steps == 0:
            #         output_dir = f"step_{completed_steps }"
            #         if config_output_dir is not None:
            #             output_dir = os.path.join(config_output_dir, output_dir)
            #         accelerator.save_state(output_dir)
            # if completed_steps >= MAX_TRAIN_STEPS:
            #     break

            # Compute train_loss and train_perplexity
            train_loss = loss.detach().float()  # total_loss.item()/(step + 1)
            training_perplexity = math.exp(train_loss)  # Not sure ????

            # Is there a spike
            if step % checkpointing_steps != 0:
                if (
                    train_loss - previous_train_loss
                ) / previous_train_loss > spike_threshold or math.isnan(train_loss):
                    spike_detected = True
                    logger.info("\n#####SPIKE_DETECTED#####")

            # Update previous_train_loss
            previous_train_loss = train_loss

            # Log every 100 steps
            # if step % 100 == 0 and step > 0 and with_tracking:
            if with_tracking:
                logger.info(
                    f"epoch {epoch} step {step}: training_perplexity: {training_perplexity} training_loss: {train_loss}\n"
                    f"Spike detected: {spike_detected}"
                )
                accelerator.log(
                    {
                        "training_perplexity": training_perplexity,
                        "train_loss": train_loss,
                        "epoch": epoch,
                        "step": completed_steps,
                        "spike": int(spike_detected),
                    },
                    step=completed_steps,
                )

            # Local checkpoint every checkpointing_steps steps
            if isinstance(checkpointing_steps, int):
                if batches:
                    batches = {
                        key: torch.cat((value, batch[key]), axis=0)
                        for key, value in batches.items()
                    }
                else:
                    batches = batch                    

                if step % checkpointing_steps == 0:

                    if config_output_dir is not None:
                        # If spikes during the checkpointing_steps last steps,
                        # save the model and the batches of the last checkpointing_steps steps into a new checkpoint
                        if step > 0:
                            if spike_detected:
                                output_dir = f"step_{step-checkpointing_steps}"
                                if config_output_dir is not None:
                                    output_dir = os.path.join(config_output_dir, output_dir)
                                accelerator.save_state(output_dir)
                                for key, value in batches.items():
                                    output_name = f"{output_dir}/{str(key)}.pt"
                                    torch.save(value, output_name)
                            else:
                                shutil.rmtree(output_dir)

                        output_dir = f"step_{step}"
                        if config_output_dir is not None:
                            output_dir = os.path.join(config_output_dir, output_dir)
                        accelerator.save_state(output_dir)

                        # Reset no_spike
                        spike_detected = False

                    #Partial evaluation after "checkpoint_steps" training steps
                    model.eval()
                    losses = []
                    print("train_dataloader length", len(train_dataloader))
                    max_eval_steps = 10 #HARD CODED FOR NOW
                    for step, batch in enumerate(eval_dataloader):
                        if step == max_eval_steps:
                            break

                        with torch.no_grad():
                            outputs = model(**batch)

                        loss = outputs.loss
                        losses.append(
                            accelerator.gather_for_metrics(loss.repeat(per_device_eval_batch_size))
                        )

                    losses = torch.cat(losses)
                    try:
                        eval_loss = torch.mean(losses)
                        eval_perplexity = math.exp(eval_loss)
                    except OverflowError:
                        eval_perplexity = float("inf")

                    logger.info(
                        f"epoch {epoch}: eval_perplexity: {eval_perplexity} eval_loss: {eval_loss}"
                        f"train_loss: {train_loss} training_perplexity: {training_perplexity}"
                    )

                    if with_tracking:
                        accelerator.log(
                            {
                                "eval_perplexity": eval_perplexity,
                                "eval_loss": eval_loss,
                                "train_loss": train_loss,
                                "training_perplexity": training_perplexity,
                                "epoch": epoch,
                                "step": completed_steps,
                            },
                            step=completed_steps,
                        )


        #Full evaluation after each epoch
        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses.append(
                accelerator.gather_for_metrics(loss.repeat(per_device_eval_batch_size))
            )

        losses = torch.cat(losses)
        try:
            eval_loss = torch.mean(losses)
            eval_perplexity = math.exp(eval_loss)
        except OverflowError:
            eval_perplexity = float("inf")

        logger.info(
            f"epoch {epoch}: eval_perplexity: {eval_perplexity} eval_loss: {eval_loss}"
            f"train_loss: {train_loss} training_perplexity: {training_perplexity}"
        )

        if with_tracking:
            accelerator.log(
                {
                    "eval_perplexity": eval_perplexity,
                    "eval_loss": eval_loss,
                    "train_loss": train_loss,
                    "training_perplexity": training_perplexity,
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )

        if checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if config_output_dir is not None:
                output_dir = os.path.join(config_output_dir, output_dir)
            accelerator.save_state(output_dir)

    if with_tracking:
        accelerator.end_training()
        accelerator.__delattr__("trackers")

    if config_output_dir is not None and os.listdir(config_output_dir) == 0:
        print("DEBUG", config_output_dir)
        os.rmdir(config_output_dir)

    # if config_output_dir is not None:
    #     accelerator.wait_for_everyone()
    #     unwrapped_model = accelerator.unwrap_model(model)
    #     unwrapped_model.save_pretrained(
    #         config_output_dir,
    #         is_main_process=accelerator.is_main_process,
    #         save_function=accelerator.save,
    #     )
    #     if accelerator.is_main_process:
    #         tokenizer.save_pretrained(config_output_dir)
    #         with open(os.path.join(config_output_dir, "all_results.json"), "w") as f:
    #             json.dump(
    #                 {
    #                     "eval_perplexity": eval_perplexity,
    #                     "training_perplexity": training_perplexity,
    #                 },
    #                 f,
    #             )


def set_up_run(
    logger, group_name, model_config, configs: Config, train_set, test_set, tokenizer_length
):
    # Rename the run
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    configs.run_name = f"{configs.run_name}_{sha}"

    accelerator_log_kwconfigs = {}

    # Initialize the accelerator. We will let the accelerator handle device placement for us.
    if configs.with_tracking:
        accelerator_log_kwconfigs["log_with"] = configs.report_to
        accelerator_log_kwconfigs["logging_dir"] = configs.output_dir

    accelerator = Accelerator(
        gradient_accumulation_steps=configs.gradient_accumulation_steps,
        mixed_precision=configs.mixed_precision,
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
    if configs.output_dir is not None:
        new_output_dir = os.path.join(configs.output_dir, f"{configs.run_name}")
        os.mkdir(new_output_dir)
        configs.output_dir = new_output_dir
    accelerator.wait_for_everyone()

    # Rewrite the model_config
    #model_config.update(configs.model_config_overwrite)

    # Load a HuggingFace (pretrained) causal language model

    if configs.model_name and configs.pretrained_model:
        logger.info("Training a pretrained model")
        model = AutoModelForCausalLM.from_pretrained(
            configs.model_name,
            config=model_config,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(
            model_config,
        )

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if tokenizer_length > embedding_size:
        model.resize_token_embeddings(tokenizer_length)

    # Log a few random samples from the training set:
    # for index in random.sample(range(len(train_set)), 3):
    #     logger.info(f"Sample {index} of the training set: {train_set[index]}.")

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
        optimizer_grouped_parameters, lr=configs.learning_rate, betas=(configs.beta_1, configs.beta_2)
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

    if configs.resume_from_checkpoint is not None:
        accelerator.load_state(configs.resume_from_checkpoint)

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
    # checkpointing_steps = configs.checkpointing_steps
    # if checkpointing_steps is not None and checkpointing_steps.isdigit():
    #     checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if configs.with_tracking:
        experiment_config = vars(configs)
        # TensorBoard cannot log Enums, need the raw value
        # experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value

        init_kwargs = (
            {
                "wandb": {
                    "group": f"{group_name}_{sha}",
                    "name": configs.run_name,
                    "settings": {"console": "off"},
                    "tags": configs.additional_tags,
                }
            }
            if group_name is not None
            else {"wandb": {"name": configs.run_name, "settings": {"console": "off"}}}
        )
        accelerator.init_trackers(
            configs.project_name,
            experiment_config,
            init_kwargs=init_kwargs,
        )

    return (
        accelerator,
        train_dataloader,
        eval_dataloader,
        model,
        optimizer,
        lr_scheduler,
        num_update_steps_per_epoch,
    )


if __name__ == "__main__":
    print("Nothing running:")
    print_gpu_utilization()

    print("Used for torch")
    torch.ones((1, 1)).to("cuda")
    print_gpu_utilization()

    params = Config()

    print("Model usage")
    config = AutoConfig.from_pretrained(params.model_name)
    model = AutoModelForCausalLM.from_config(config)
    print_gpu_utilization()
