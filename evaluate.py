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
from model_zoo.utils import TrainerWithLogger
import argparse
seed=42
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
# torch seed init.
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--dataset', type=str,default="c4")

    return parser.parse_args()

args = parse_args()

with open(args.config, 'r') as f:
    cfg = yaml.safe_load(f)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B")
if cfg['model_type'] == "Qwen3MoELPR":
    config = Qwen3MoeConfig_LPR.from_pretrained(args.checkpoint_path)
    model = Qwen3MoeForCausalLM_LPR.from_pretrained(args.checkpoint_path,config=config)
    num_experts = num_experts = cfg["model_config"]["num_experts"]
elif cfg['model_type'] == "Qwen3MoE":
    config = Qwen3MoeConfig.from_pretrained(args.checkpoint_path)
    model = Qwen3MoeForCausalLM.from_pretrained(args.checkpoint_path,config=config)
    num_experts = num_experts = cfg["model_config"]["num_experts"]
elif cfg['model_type'] == "DeepseekV3":
    config = DeepseekV3Config.from_pretrained(args.checkpoint_path)
    model = DeepseekV3ForCausalLM.from_pretrained(args.checkpoint_path,config=config)
    num_experts = cfg["model_config"]["n_routed_experts"]
elif cfg['model_type'] == "DeepseekV3LPR":
    config = DeepseekV3Config_LPR.from_pretrained(args.checkpoint_path)
    model = DeepseekV3ForCausalLM_LPR.from_pretrained(args.checkpoint_path,config=config)
    num_experts = cfg["model_config"]["n_routed_experts"]
elif cfg['model_type'] == "Llama4":
    config = Llama4Config.from_pretrained(args.checkpoint_path)
    model = Llama4ForCausalLM.from_pretrained(args.checkpoint_path,config=config)
elif cfg['model_type'] == "Mixtral":
    num_experts = num_experts = cfg["model_config"]["num_local_experts"]
    config = MixtralConfig.from_pretrained(args.checkpoint_path)
    model = MixtralForCausalLM.from_pretrained(args.checkpoint_path,config=config)
elif cfg['model_type'] == "MixtralLPR":
    num_experts = num_experts = cfg["model_config"]["num_local_experts"]
    config = MixtralConfig_LPR.from_pretrained(args.checkpoint_path)
    model = MixtralForCausalLM_LPR.from_pretrained(args.checkpoint_path,config=config)
else:
    raise NotImplementedError()

tokenizer.pad_token = tokenizer.eos_token

from datasets import load_dataset

eval_dataset = load_dataset("json", 
                        # streaming=True,
                        data_files={"validation": "/root/autodl-tmp/DDMoE/data/c4val/c4-validation.*-of-00008.json.gz"})
eval_dataset = eval_dataset["validation"]
eval_dataset = eval_dataset.select(range(int(len(eval_dataset))))

def tokenize_fn(example):

    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

eval_dataset = eval_dataset.map(tokenize_fn, batched=True)

max_length = cfg["training_config"]["max_length"]

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=max_length)

eval_dataset = eval_dataset.map(tokenize_fn, batched=True,
        remove_columns=["text", "timestamp", "url"]  )

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir=f"./checkpoints/{cfg['model_name']}/eval",
    per_device_eval_batch_size=25,
    do_eval=True,
    fp16=True,
    remove_unused_columns=True,  
    dataloader_prefetch_factor=2,
    dataloader_pin_memory=True,
    dataloader_num_workers=25,  
    disable_tqdm=False,
    log_level="critical"
)
collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=None,
    return_tensors="pt"
)
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=eval_dataset,
    data_collator=collator
)
with torch.no_grad():
    metrics = trainer.evaluate()
    print(metrics)
    eval_loss = metrics['eval_loss']
    ppl = math.exp(eval_loss)
    print(f"Validation Loss: {eval_loss:.4f} | Perplexity: {ppl:.2f}")