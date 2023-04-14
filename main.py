import argparse
import os
from accelerate.logging import get_logger
from src.utils import convert_yaml_configs, load_dataset_for_training, train, set_up_run


#Instanciate the logger
logger = get_logger(__name__)
os.environ['WANDB_START_METHOD'] = 'thread'

parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
parser.add_argument(
    "--lr",
    type=float,
    default=None,
    help="Initial learning rate (after the potential warmup period) to use.",
)
parser.add_argument(
    "--group_name",
    type=str,
    default=None,
    help="Group name of the run in wandb",
)
args = parser.parse_args()


#Get the configs
configs = convert_yaml_configs()
configs = convert_yaml_configs(
    overrides={
        "learning_rate": args.lr,
        "run_name": f"LR_{args.lr}",
    }
) if args.lr is not None else  convert_yaml_configs()


# Load the dataset:
train_set, test_set, tokenizer_length, model_config = load_dataset_for_training(
    data_location=configs.data_location,
    train_data_location=configs.train_data_location,
    test_data_location=configs.test_data_location,
    debug_set=configs.debug_set,
    train_debug_set_location=configs.train_debug_set_location,
    test_debug_set_location=configs.test_debug_set_location,
    tokenizer_name=configs.tokenizer_name,
    model_name=configs.model_name,
    use_slow_tokenizer=configs.use_slow_tokenizer,
    model_config_name=configs.model_config_name,
    dataset_name=configs.dataset_name,
    dataset_config_name=configs.dataset_config_name,
    validation_split_percentage=configs.validation_split_percentage,
    preprocessing_num_workers=configs.preprocessing_num_workers,
    overwrite_cache=configs.overwrite_cache,
    logger=logger,
    block_size=configs.block_size,
    seed=configs.seed,
)

(
    accelerator,
    train_dataloader,
    eval_dataloader,
    model,
    optimizer,
    lr_scheduler,
    num_update_steps_per_epoch,
) = set_up_run(
    logger=logger,
    group_name=args.group_name,
    model_config=model_config,
    configs=configs,
    train_set=train_set,
    test_set=test_set,
    tokenizer_length=tokenizer_length,
)

train(
    logger=logger,
    accelerator=accelerator,
    train_set=train_set,
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader,
    model=model,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    num_update_steps_per_epoch=num_update_steps_per_epoch,
    per_device_train_batch_size=configs.per_device_train_batch_size,
    gradient_accumulation_steps=configs.gradient_accumulation_steps,
    num_train_epochs=configs.num_train_epochs,
    max_train_steps=configs.max_train_steps,
    resume_from_checkpoint=configs.resume_from_checkpoint,
    with_tracking=configs.with_tracking,
    spike_threshold=configs.spike_threshold,
    checkpointing_steps=configs.checkpointing_steps,
    config_output_dir=configs.output_dir,
    per_device_eval_batch_size=configs.per_device_eval_batch_size,
)
