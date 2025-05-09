import os
os.environ['HF_HOME'] = '[mypath]' + '/.cache/huggingface'
import sys
from pathlib import Path
import hydra
import torch
import datasets
from transformers import  AutoTokenizer, AutoConfig, GPTNeoForCausalLM, AutoModelForCausalLM
import torch
import time
import logging
from trainer_decoupled import DecoupledTrainer

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

logger = logging.getLogger("distributed_worker")

os.environ['HYDRA_FULL_ERROR'] = str(1)


global _tqdm_active 
_tqdm_active = False


@hydra.main(config_path="./config", config_name="config.yaml", version_base=None)
def main(cfg):

    torch.cuda.manual_seed_all(42)  # if you are using multi-GPU        
    root_path_model = '[mypath]' + '/HuggingFace_Models/'
    root_path_data = '[mypath]' + '/HuggingFace/'
    

    if cfg.train.finetune:
        #load pretrain model
        model = AutoModelForCausalLM.from_pretrained(root_path_model+ cfg.model.config_path)
    else:
        #Instantiate new model using config file
        # config
        model_config = AutoConfig.from_pretrained(os.getcwd() + cfg.model.config_path)
        # 1. create model
        model = GPTNeoForCausalLM(model_config)
        
    #model = torch.compile(model)
    print('model instantiated')
    tokenizer = AutoTokenizer.from_pretrained(root_path_model + cfg.model.tokenizer)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print('tokenizer loaded')
    # 3. load dataset
    dataset = datasets.load_dataset(cfg.data.path)
    dataset = dataset['train'].train_test_split(0.05, seed = 42)
    
    logger.info('DDP')
    
    trainer = DecoupledTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset['train'],
        eval_dataset =  dataset['test'],
        text_column_name  = 'text',
        args = cfg.train,
        log = logger,
        preprocess_dataset_fn = None,
        run_name = cfg.run_name
    )
    
    # launch the client training process
    trainer.train()

if __name__=="__main__":
    
    main()
