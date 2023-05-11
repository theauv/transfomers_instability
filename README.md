# Transformers_instability

This project aims at exploring the instabilities in large language modeling transformers training.
We first want to reproduce such behaviors with a moderate size model and see how this can be solved.

## Installation

Create an environment from the requirements file

```bash
conda create --name transformers_instability --file requirements.txt
```

## Overview

#### The files

- **configs**: folder constaining all the config files to run the model (yaml files)
- **src**:
    - **config_dataclass.py**: dataclass gathering all the configs. It is used to initialize the model etc..
    - **utils.py**: contains all the different functions used by the main process (sorry, it is not very modularized and readable at the moment)
- **main.py**: pipeline gathering all the building blocks (read the configs, download the dataset, set up the huggingface accelerator and wandb, train the model)

#### Libraries and API

- **Dataset and model**: The main libraries used are *datasets* and *transformers* from huggingface. It is used to define the model and the dataset.
- **Training process**: For the training process and linking between the different part of the code we use *accelerate*
- **Render and plots**: We use *wandb* as it is very easy to use with *accelerate* and *huggingface* in general.

## Usage

#### 1. Change the parameters

In the configs directory, you can change the values of the parameters (batch size, learning rate, ...) in the different yaml config files:
- **trainer.yaml**: contains everything about the trainer (learning rate, weight decay, ...)
- **model.yaml**: contains everything about the model (model_name, ...)
- **dataset.yaml**: contains everything about the dataset (dataset name, split, tokenizer, ...)
- **utils.yaml**: contains all sort of utils things (run_name, report_to, ...)

#### 2. Run the main pipeline

The main.py is at the moment the main pipeline used to train the model.
It basically:
- Downloads the dataset from the hub (or reload it from local files if saved)
- Instantiates the model specified in config.py
- Instantiates a tokenizer
- Preprocesses the dataset
- Trains the model
- saves local checkpoint if there are spikes
- send the logs to wandb to give cool training plots

You can access the logs and charts in "wandb".
