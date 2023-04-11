# Transformers_instability

This project aims at exploring the instabilities in large language modeling transformers training.
We first want to reproduce such behaviors with a moderate size model and see how this can be solved.

Sorry, for the moment the repo is a bit messy as it contains unused files that might be used depending on how goes the project.

## Installation

Create an environment from the requirements file

```bash
conda create --name transformers_instability --file requirements.txt
```

## Usage

1. Change the parameters

In the configs directory, you can change the values of the parameters (batch size, learning rate, ...) in the different yaml config files

2. Run the main pipeline

The hf_transformer_no_trainer.py is at the moment the main pipeline used to train the model.
It basically:
- Download the dataset from the hub (or reload it from local files if saved)
- Instantiate the model specified in config.py
- Instantiate a tokenizer
- Preprocess the dataset
- Train the model
- local checkpoint if there are spikes

You can access the logs and charts in "wandb".

## Next steps

- Train to reach better perplexity
- Increase batch_size without memory exceeded
- Grid search with different learning rates (issues with VM when vs code disconnected)
- smaller model ?
