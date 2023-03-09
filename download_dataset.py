from datasets import load_dataset
from transformers import AutoTokenizer


dataset = load_dataset("super_glue", 'boolq')

print(dataset)
print(type(dataset))
dataset['train'].to_json('data_and_tokenizers/data/train/train_data.json')
dataset['test'].to_json('data_and_tokenizers/data/test/test_data.json')
dataset['validation'].to_json('data_and_tokenizers/data/valid/valid_data.json')

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer.save_pretrained("data_and_tokenizers/tokenizers/")