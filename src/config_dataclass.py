from dataclasses import dataclass
from typing import Union
from transformers import SchedulerType

# YOU DONT NEED TO MODIFY THS, HAVE A LOOK AT THE CONFIGS YAML FILES


@dataclass
class Config:
    project_name: str = "transformers_instability_no_trainer"
    run_name: str = "test"
    data_location: str = "/home/theau/transformers_instability/data_and_tokenizers/data"
    train_data_location: str = f"{data_location}/train/train_data.json"
    test_data_location: str = f"{data_location}/valid/test_data.json"
    dataset_name: str = "wikitext"
    dataset_config_name: str = "wikitext-2-raw-v1"
    validation_split_percentage: float = 0.8
    model_name: str = "distilgpt2"
    model_config_name: Union[str, None] = None
    pretrained_model: bool = True
    tokenizer_name: Union[str, None] = None
    use_slow_tokenizer: bool = False
    preprocessing_num_workers: Union[None, int] = None
    overwrite_cache: bool = False
    block_size: Union[int, None] = 128  # None
    per_device_eval_batch_size: int = 4  # 32
    per_device_train_batch_size: int = 4  # 32
    weight_decay: float = 0.0
    learning_rate: float = 1e-2
    gradient_accumulation_steps: int = 1
    num_train_epochs: int = 5
    max_train_steps: Union[None, int] = None
    lr_scheduler_type: SchedulerType = "linear"
    num_warmup_steps: float = 10  # 5e3
    checkpointing_steps: Union[None, str, int] = "step"
    resume_from_checkpoint: Union[None, str] = None
    with_tracking: bool = True
    output_dir: str = "outputs"
    report_to: str = "wandb"
    seed: int = 0
    spike_threshold: float = 10.0
    debug_set: bool = False
    train_debug_set_location: str = "train/train_debug_set.json"
    test_debug_set_location: str = "valid/test_debug_set.json"
