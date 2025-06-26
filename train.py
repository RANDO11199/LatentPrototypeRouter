import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
import yaml
import math
from transformers import get_wsd_schedule
from model_zoo.Qwen3Moe import Qwen3MoeForCausalLM_LPR, Qwen3MoeConfig_LPR
from model_zoo.DeepSeekV3 import DeepseekV3ForCausalLM_LPR, DeepseekV3Config_LPR
from model_zoo.Mixtral import MixtralForCausalLM_LPR, MixtralConfig_LPR
from transformers import (Qwen3MoeForCausalLM,
                          DeepseekV3ForCausalLM,
                          DeepseekV3Config,
                          Qwen3MoeConfig,
                          Llama4ForCausalLM,
                          Llama4Config,
                          MixtralForCausalLM,
                          MixtralConfig)
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import AdamW
import random
import numpy as np
from model_zoo.utils import *
import argparse
seed=42
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
# torch seed init.
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def train_pipeline():
    ''''''
    args = parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    total_steps = cfg["training_config"]["total_steps"]
    warmup_steps = int(total_steps * cfg["training_config"]["warmup_ratio"])
    stable_steps = int(total_steps * cfg["training_config"]["stable_ratio"])
    max_length = cfg["training_config"]["max_length"]
    decay_steps = total_steps - warmup_steps - stable_steps
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B")
    tokenizer.pad_token = tokenizer.eos_token
    def tokenize_function(examples):
        ''''''
        return tokenizer(examples["text"], truncation=True, max_length=max_length)
    
    def load_preprocessed_data():
        ''''''
        dataset = load_dataset("parquet", 
                               streaming=True,
                               data_files={"train": "/data/fineweb/sample/100BT/*.parquet",
                                            })
        test_set = load_dataset("json", 
                               streaming=True,
                               data_files={"validation": "/data/c4val/c4-validation.*-of-00008.json.gz"})
        return dataset, test_set
    
    # 加载训练集和验证集
    ds, test_ds = load_preprocessed_data()

    tokenized_train_dataset = ds.map(
        tokenize_function,
        batched=True,
        remove_columns=["text", "timestamp", "url"]  # 移除无用字段
    )
    tokenized_test_dataset = test_ds.map(
        tokenize_function,
        batched=True,
        remove_columns=["text", "timestamp", "url"]  # 移除无用字段
    )

    # 初始化模型
    if cfg['model_type'] == "Qwen3MoELPR":
        model = Qwen3MoeForCausalLM_LPR(Qwen3MoeConfig_LPR(**cfg["model_config"]))
        num_experts = num_experts = cfg["model_config"]["num_experts"]
    elif cfg['model_type'] == "Qwen3MoE":
        model = Qwen3MoeForCausalLM(Qwen3MoeConfig(**cfg["model_config"]))
        num_experts = num_experts = cfg["model_config"]["num_experts"]
    elif cfg['model_type'] == "DeepseekV3":
        model = DeepseekV3ForCausalLM(DeepseekV3Config(**cfg["model_config"]))
        num_experts = cfg["model_config"]["n_routed_experts"]

    elif cfg['model_type'] == "DeepseekV3LPR":
        model = DeepseekV3ForCausalLM_LPR(DeepseekV3Config_LPR(**cfg["model_config"]))
        num_experts = cfg["model_config"]["n_routed_experts"]
    elif cfg['model_type'] == "Llama4":
        model = Llama4ForCausalLM(Llama4Config(**cfg["model_config"]))
    elif cfg['model_type'] == "Mixtral":
        num_experts = num_experts = cfg["model_config"]["num_local_experts"]
        model = MixtralForCausalLM(MixtralConfig(**cfg["model_config"]))
    elif cfg['model_type'] == "MixtralLPR":
        num_experts = num_experts = cfg["model_config"]["num_local_experts"]
        model = MixtralForCausalLM_LPR(MixtralConfig_LPR(**cfg["model_config"]))
    else:
        raise NotImplementedError()

    model.resize_token_embeddings(len(tokenizer))  
    model = model.to("cuda")  
    total_params = count_parameters(model) / 1e9 
    print(f"Total Parameter: {total_params}B")

    args = TrainingArguments(
        output_dir=f"./checkpoints/{cfg['model_name']}",
        logging_dir="./logs/" + cfg['model_name'],       
        logging_steps=1,          
        report_to="tensorboard",            
        logging_strategy="steps",   
        per_device_train_batch_size=cfg["training_config"]["per_device_train_batch_size"], 
        per_device_eval_batch_size = cfg["training_config"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["training_config"]["gradient_accumulate_steps"],
        max_grad_norm= cfg["training_config"]["gradient_clip_norm"],
        fp16=True,
        gradient_checkpointing= True,
        log_level="info",
        max_steps=total_steps,
        save_steps=500,
        remove_unused_columns=True,  
        dataloader_prefetch_factor=2, 
        dataloader_pin_memory=True,
        dataloader_num_workers=16,  
    )
    

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=None,
        return_tensors="pt"
    )
    optimizer = AdamW(model.parameters(),
                      lr = cfg["training_config"]["lr"],
                      weight_decay= cfg["training_config"]["weight_decay"],
                      betas = (cfg["training_config"]["adam_beta1"],cfg["training_config"]["adam_beta2"]))

    lr_scheduler = get_wsd_schedule(
        optimizer=optimizer, 
        num_warmup_steps=warmup_steps,
        num_stable_steps= stable_steps,
        num_decay_steps= decay_steps,
        min_lr_ratio = cfg["training_config"]["min_lr_ratio"]
    )

    trainer = TrainerWithLogger(
        router_aux_loss_coef = cfg["model_config"]["router_aux_loss_coef"],
        router_kl_loss_coef = cfg["model_config"]["router_kl_loss_coef"],
        num_experts = num_experts,
        num_hidden_layers =  cfg["model_config"]["num_hidden_layers"],
        top_k = cfg["model_config"]["num_experts_per_tok"],
        model=model,
        args=args,
        optimizers= (optimizer,lr_scheduler),
        train_dataset=tokenized_train_dataset['train'],
        eval_dataset=tokenized_test_dataset['validation'],
        data_collator=collator,
        model_type = cfg['model_type']
    )
    trainer.train()
    if cfg["eval"]:
        with torch.no_grad():
            eval_result = trainer.evaluate()
            eval_loss = eval_result['eval_loss']
            ppl = math.exp(eval_loss)

        print(f"Validation Loss: {eval_loss:.4f} | Perplexity: {ppl:.2f}")
        trainer.writer.add_scalar(f"Perplexity", ppl)
        trainer.writer.add_scalar(f"eval_loss", eval_loss)
if __name__ == "__main__":
    train_pipeline()
