from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    AdamW,
    # RobertaConfig,
    # RobertaModel,
    AutoModelForCausalLM,
    GPT2Model,
    GPT2Config,
)
import evaluate
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import wandb
import git



class CustomTrainer(Trainer):
    def create_optimizer(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.1,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            betas=(0.9, 0.95),
        )

        return self.optimizer


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def preprocess_function(examples):
    return tokenizer([" ".join(x) for x in examples["answers.text"]])


def group_texts(examples, block_size=128):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


if __name__ == "__main__":
    # Download/Reload the dataset
    DATA_LOCATION = "/home/theau/transformers_instability/data_and_tokenizers/data"
    TRAIN_DATASET_LOCATION = f"{DATA_LOCATION}/train/train_data.json"
    TEST_DATASET_LOCATION = f"{DATA_LOCATION}/valid/test_data.json"
    if Path(TRAIN_DATASET_LOCATION).is_file() and Path(TEST_DATASET_LOCATION).is_file():
        train_set = load_dataset("json", data_files=TRAIN_DATASET_LOCATION)["train"]
        test_set = load_dataset("json", data_files=TEST_DATASET_LOCATION)["train"]
    else:
        dataset = load_dataset("eli5", split="train_asks[:5000]")
        dataset = dataset.train_test_split(test_size=0.2)
        dataset = dataset.flatten()

        # Tokenize the dataset
        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

        tokenized_dataset = dataset.map(
            preprocess_function,
            batched=True,
            num_proc=4,
            remove_columns=dataset["train"].column_names,
        )

        lm_dataset = tokenized_dataset.map(group_texts, batched=True, num_proc=4)

        train_set = lm_dataset["train"].shuffle(seed=42)
        train_set.to_json(TRAIN_DATASET_LOCATION)
        test_set = lm_dataset["test"].shuffle(seed=42)
        test_set.to_json(TEST_DATASET_LOCATION)

        #Data Collator
        # tokenizer.pad_token = tokenizer.eos_token
        # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Load a HuggingFace pretrained causal language model
    # configuration = RobertaConfig()
    # model = RobertaModel(configuration)
    config = GPT2Config()
    model = AutoModelForCausalLM.from_config(config)

    #Name of the run
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha


    RUN_NAME = input("Give a name to the run \n")
    RUN_NAME = "test" if RUN_NAME == "" else RUN_NAME
    RUN_NAME = f"{sha}_{RUN_NAME}"
    print(f"Name fo the RUN: {RUN_NAME}")

    training_args = TrainingArguments(
        output_dir="results",  # output directory
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        eval_steps=1,
        num_train_epochs=20,  # total # of training epochs
        per_device_train_batch_size=32,  # batch size per device during training
        per_device_eval_batch_size=32,  # batch size for evaluation
        warmup_steps=5000,  # number of warmup steps for learning rate scheduler
        learning_rate=1e-2, #2e-5,
        weight_decay=0.,  # strength of weight decay
        logging_dir="logs",  # directory for storing logs
        report_to="wandb",  # enable logging to W&B
        run_name=RUN_NAME,  # name of the W&B run (optional)
        max_grad_norm=1.0,
        eval_accumulation_steps=20,
        prediction_loss_only=True,
        logging_steps=100,
        save_steps=100,
    )
    metric = evaluate.load("accuracy")

    # Create a trainer class and train the model
    # os.environ["WANDB_DISABLED"] = "true"

    print(f"Cuda available: {torch.cuda.is_available}")
    torch.cuda.empty_cache()

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=test_set,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    wandb.finish()
