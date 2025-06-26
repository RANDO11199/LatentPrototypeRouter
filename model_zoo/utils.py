from torch.utils.tensorboard import SummaryWriter
from transformers import Trainer
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)
from transformers.trainer import _is_peft_model
import torch
from transformers import TrainerCallback
import torch.nn as nn
import torch.nn.functional as F

import csv
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    return parser.parse_args()

def get_activation_size(model, input_shape=(1, 512)):

    model.eval()
    dummy_input = torch.randint(0,10000,input_shape).to(model.device)
    
    # Hook捕获各层输出尺寸
    activation_sizes = []
    def hook_fn(module, input, output):
        activation_sizes.append(output.element_size() * output.nelement())
    
    hooks = []
    for layer in model.modules():
        if isinstance(layer, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.MultiheadAttention)):
            hooks.append(layer.register_forward_hook(hook_fn))
    
    with torch.no_grad():
        model(dummy_input)
    
    # 移除hooks
    for hook in hooks:
        hook.remove()
    
    return sum(activation_sizes) / (1024**3)  # 转换为GB

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class LoadBalancer(nn.Module):
    def __init__(self, num_experts, model_type,decay=0.99,top_k=8, device="cuda"):
        super().__init__()
        self.num_experts = num_experts
        self.decay = decay
        self.device = device
        self.top_k = top_k
        # 注册缓冲区
        self.register_buffer("hard_load", torch.zeros(num_experts, device=device))
        self.register_buffer("soft_load", torch.zeros(num_experts, device=device))
        self.register_buffer("load_mean", torch.zeros(num_experts, device=device))
        self.register_buffer("load_m2", torch.zeros(num_experts, device=device))  
        self.register_buffer("count", torch.tensor(0, device=device))
        self.model_type = model_type

    @torch.no_grad()
    def get_topk_indices(self, scores):
        scores_for_choice = scores.view(-1, self.n_routed_experts) + self.e_score_correction_bias.unsqueeze(0)
        group_scores = (
            scores_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        return topk_indices
    
    def update(self, scores):

        decay = self.decay

        if self.model_type=="DeepseekV3":
            topk_idx = scores
        else:
            batch_size = scores.size(0)
            topk_vals, topk_idx = torch.topk(scores, self.top_k, dim=-1)
            routing_weights = F.softmax(topk_vals, dim=-1, dtype=torch.float) 
            full_weights = torch.zeros(batch_size, self.num_experts, device=self.device)
            full_weights.scatter_(1, topk_idx, routing_weights)
            soft_load = torch.zeros(self.num_experts, device=self.device)
            # 2. 软负载（路由权重总和）
            soft_load = full_weights.sum(dim=0)
            self.soft_load.mul_(decay).add_(soft_load, alpha=1-decay)
            delta = soft_load - self.load_mean
            self.load_mean.add_(delta / self.count)
            delta2 = soft_load - self.load_mean
            self.load_m2.add_(delta * delta2)
        hard_load = torch.zeros(self.num_experts, device=self.device)

        hard_load.scatter_add_(0, topk_idx.flatten(), torch.ones(topk_idx.numel(), device=self.device))
        # 3. EMA更新
        self.hard_load.mul_(decay).add_(hard_load, alpha=1-decay)
        # 4. 在线计算负载方差（Welford算法）
        self.count += 1
    
    @property
    def variance(self):
   
        return self.load_m2 / self.count.clamp(min=1)
    
    @property
    def normalized_hard_load(self):

        total = self.hard_load.sum().clamp(min=1e-6)
        return self.hard_load / total
    
    @property
    def normalized_soft_load(self):

        total = self.soft_load.sum().clamp(min=1e-6)
        return self.soft_load / total
    
    def compute_imbalance_metrics(self, load_vector):

        valid_load = load_vector
        n = len(valid_load)
        
        if n < 2:
            return {
                "cv": 0.0,
                "min_max_ratio": 0.0,
                "gini": 0.0
            }
        
  
        mean = valid_load.mean()
        std = valid_load.std()
        cv = std / mean
        

        min_val = valid_load.min()
        max_val = valid_load.max()
        min_max_ratio = min_val / max_val
        
      
        sorted_load = torch.sort(valid_load).values
        index = torch.arange(1, n+1, device=self.device)
        gini = torch.sum((2 * index - n - 1) * sorted_load) / (n * torch.sum(sorted_load))
        
        return {
            "cv": cv.item(),
            "min_max_ratio": min_max_ratio.item(),
            "gini": gini.item()
        }
    
    def get_metrics(self):

        hard_metrics = self.compute_imbalance_metrics(self.normalized_hard_load)
        soft_metrics = self.compute_imbalance_metrics(self.normalized_soft_load)
        
        return {
            "hard_load": self.hard_load.clone(),
            "soft_load": self.soft_load.clone(),
            "variance": self.variance.mean().item(),
            "hard_cv": hard_metrics["cv"],
            "hard_gini": hard_metrics["gini"],
            "soft_cv": soft_metrics["cv"],
            "soft_gini": soft_metrics["gini"],
            "hard_max_min":  hard_metrics["min_max_ratio"],
            "soft_max_min":  soft_metrics["min_max_ratio"],
            "utilization": (self.hard_load > 0).float().mean().item()
        }


class TrainerWithLogger(Trainer):
    def __init__(self,router_aux_loss_coef,router_kl_loss_coef, num_experts, num_hidden_layers,top_k,model_type, *arg,**kwargs):
        super().__init__(*arg,**kwargs)
        self.model_type = model_type
        self.router_aux_loss_coef = router_aux_loss_coef
        self.router_kl_loss_coef = router_kl_loss_coef
        self.num_hidden_layers = num_hidden_layers
        self.writer = SummaryWriter(log_dir=self.args.logging_dir)
        self.loadbalancer = [LoadBalancer(num_experts,top_k=top_k,model_type=model_type) for _ in range(num_hidden_layers)]
        self.counter = 0
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}
        outputs = model(**inputs)
        try:
            net_loss = outputs.net_loss
        except:
            net_loss = 0.
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            # User-defined compute_loss function
            if self.compute_loss_func is not None:
                loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes
        try:
            aux_loss = outputs["aux_loss"].item()*self.router_aux_loss_coef
        except:
            aux_loss = torch.tensor(0.)
        try:

            kl_loss = outputs["kl_loss"].item()*self.router_kl_loss_coef
        except:
            kl_loss = torch.tensor(0.)

        # net_loss = outputs["loss"].item() - kl_loss - aux_loss
        step = self.state.global_step
        self.writer.add_scalar("train/aux_loss", aux_loss , step)
        self.writer.add_scalar("train/kl_loss", kl_loss, step)
        self.writer.add_scalar("train/net_loss", net_loss, step)
        for idx in range(self.num_hidden_layers):
            if self.model_type=="Qwen3MoE" or "Qwen3MoELPR" or "DeepseekV3LPR":
                router_logit = outputs['router_logits'][idx].detach()
            elif self.model_type=="DeepseekV3":
                router_logit = outputs['topk_indices'][idx].detach()
            self.loadbalancer[idx].update(router_logit)
            if self.counter % 100 == 0:
                Metrics = self.loadbalancer[idx].get_metrics()
                GINI = Metrics["hard_gini"]
                # hard_load = Metrics["hard_load"]
                variance = Metrics["variance"]
                hard_cv = Metrics["hard_cv"]
                utilization = Metrics["utilization"]
                hard_min_max = Metrics["hard_max_min"]
                distb = self.loadbalancer[idx].normalized_hard_load
                
                load_vector = distb.cpu().tolist()
                prefix = "expert_load"

                scalar_dict = {
                    f"layer_{idx}/expert_{i}": load for i, load in enumerate(load_vector)
                }
                self.writer.add_scalars(main_tag=prefix, tag_scalar_dict=scalar_dict, global_step=step)
     
                self.writer.add_scalar(f"layer_{idx}/Variance", variance, step)
                self.writer.add_scalar(f"layer_{idx}/GINI", GINI, step)
                self.writer.add_scalar(f"layer_{idx}/CV", hard_cv, step)
                self.writer.add_scalar(f"layer_{idx}/Utilization", utilization, step)
                self.writer.add_scalar(f"layer_{idx}/hard_min_max", hard_min_max, step)
                self.writer.add_histogram(f'layer_{idx}_load/hist', distb, step)
        
        self.counter+=1
        return (loss, outputs) if return_outputs else loss

    # def training_step(self, model, inputs):
    
    #     loss = super().training_step(model, inputs)

    #     for module in model.modules():
    #         print('emaupdate')
    #         if hasattr(module, "update_expert_keys") and callable(module.update_expert_keys):
    #             module.update_expert_keys()

    #     return loss