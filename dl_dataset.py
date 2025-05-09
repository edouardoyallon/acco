import os

import hydra
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

@hydra.main(config_path="./config", config_name="config")
def main(cfg):
    def tokenize_data_const_len(element):
        #Concat all text id seperated with eos id
        #return sequences of fixed size (discard last seq if len()< max)
        outputs = tokenizer(element['text'], truncation=False)
        input_batch = []
        input_ids_concat = []
        context_length = cfg.train.args.max_length
        for input_ids in outputs["input_ids"]:
            input_ids.append(tokenizer.eos_token_id)
            input_ids_concat+=input_ids
        batch_size = len(input_ids_concat)//context_length
        input_batch = torch.tensor(input_ids_concat[:batch_size*context_length]).reshape(batch_size, context_length)
        # no pad => no attention_mask
        return {"input_ids": input_batch }

    
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer)
    data_save_path = 'MY_PATH'
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    dataset = load_dataset(cfg.data.path)
    dataset = dataset['train'].train_test_split(0.05, seed = 42)
    dataset = dataset.map(tokenize_data_const_len, batched=True, remove_columns=dataset.column_names, num_proc=16)
    dataset.save_to_disk(data_save_path)
    print(f"Dataset saved to {data_save_path}")


if __name__ == "__main__":
    main()
