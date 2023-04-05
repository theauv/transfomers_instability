from pynvml import *
import torch
from transformers import AutoModelForCausalLM, AutoConfig
from config import Config


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()


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