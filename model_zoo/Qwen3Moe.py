import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import  List, Optional, Tuple, Union
from functools import partial
from transformers import (
    Qwen3MoeConfig,
    Qwen3MoeSparseMoeBlock,
    Qwen3MoeMLP,
    Qwen3MoeModel,
    Qwen3MoeDecoderLayer,
    Qwen3MoeForCausalLM,
)
from transformers.cache_utils import DynamicCache
from transformers.processing_utils import Unpack
from transformers.utils import logging, LossKwargs, can_return_tuple
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from .router import TokenDistributionRouter
from transformers.modeling_outputs import ( MoeCausalLMOutputWithPast,
                                            MoeModelOutputWithPast)
from dataclasses import dataclass

logger = logging.get_logger(__name__)
class Qwen3MoeConfig_LPR(Qwen3MoeConfig):
    model_type = "qwen3_moe_lpr"
    def __init__(
        self,
        output_router_kl=False,
        router_kl_loss_coef=0.001,
        unit_ball = False,
        kl_weight = 0.01,
        align_weight = 0.05,
        div_weight = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.output_router_kl = output_router_kl ###
        self.router_kl_loss_coef = router_kl_loss_coef ###
        self.unit_ball = unit_ball
        self.kl_weight = kl_weight
        self.align_weight = align_weight
        self.div_weight = div_weight


class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...

# Modification 1: add kl losses entry and net loss entry ( for recording)
@dataclass
class MoeCausalLMOutputWithPast_KL(MoeCausalLMOutputWithPast):
    kl_loss: Optional[Tuple[torch.FloatTensor]] = None
    net_loss: Optional[torch.FloatTensor] = None
# Modification 2: add kl losses entry
@dataclass
class MoeModelOutputWithPast_KL(MoeModelOutputWithPast):
    kl_losses: Optional[Tuple[torch.FloatTensor]] = None


class Qwen3MoeSparseMoeBlock_LPR(Qwen3MoeSparseMoeBlock):
    def __init__(self, config,layer_idx):
        super().__init__(config,layer_idx)
        self.gate = TokenDistributionRouter(config,layer_id=layer_idx) # Modification 3: replace the vanilla router
        self.global_step = 0
    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_values=None, output_attentions=False, output_hidden_states=False, return_dict=True):
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        routing_weights, kl, selected_experts, router_logits, z_decoed = self.gate(hidden_states) # Modification 4: match the output of LPR 
        hidden_states = hidden_states + z_decoed 

        if self.norm_topk_prob: 
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
          
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim) 
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None] 

            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype)) 
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits, kl # Modification 5: lpr regulation loss

class Qwen3MoeDecoderLayer_LPR(Qwen3MoeDecoderLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
     
        if (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen3MoeSparseMoeBlock_LPR(config,layer_idx) # Modification 6: replace the modified SparseMoEBlock
        else:
            self.mlp = Qwen3MoeMLP(config, intermediate_size=config.intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  
        output_router_kl: Optional[bool] = False, # Modification 7: lpr regulation loss
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
   
        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states
        
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states, router_logits, kl = hidden_states # Modification 8: lpr regulation loss
        else:
            router_logits = None

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if output_router_logits:
            outputs += (router_logits,)
        # Modification 9: lpr regulation loss
        if output_router_kl:
            outputs += (kl,)
        return outputs

class Qwen3MoeModel_LPR(Qwen3MoeModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen3MoeDecoderLayer`]

    Args:
        config: Qwen3MoeConfig
    """

    def __init__(self, config: Qwen3MoeConfig_LPR):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [Qwen3MoeDecoderLayer_LPR(config, layer_idx) for layer_idx in range(config.num_hidden_layers)] # Modification 10: use modified DecoderLayer
        )
        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_router_kl: Optional[bool] = None, # Modification 11: lpr regulation loss
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> MoeModelOutputWithPast_KL:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_router_kl = (output_router_kl if output_router_kl is not None else self.config.output_router_kl
                            ) # Modification 12: lpr regulation loss
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        all_router_kl = () if output_router_kl else None # Modification 13: lpr regulation loss
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    partial(decoder_layer.__call__, **flash_attn_kwargs),
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    output_router_logits,
                    use_cache,
                    cache_position,
                    position_embeddings,
                    output_router_kl, # Modification 13: lpr regulation loss
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    output_router_kl = output_router_kl, # Modification 14: lpr regulation loss
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_router_logits:
                all_router_logits += (layer_outputs[1],) # Modification 15:change to layer_outputs[1],  TODO: what if output_attentions?
            if output_router_kl:
                all_router_kl += (layer_outputs[2],) # Modification 16: lpr regulation loss

        hidden_states = self.norm(hidden_states)


        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return MoeModelOutputWithPast_KL(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
            kl_losses=all_router_kl # Modification 17: lpr regulation loss
        )

import math
def metric_aware_load_balance(scores: torch.Tensor,
                              num_experts: Optional[int] = None,

                              mode: str = "mse") -> torch.Tensor:
    """
    scores:            [B, M] raw routing logits (neg. distance or cos-sim)
    routing_weights:   [B, M] softmaxed affinities p_{t,i}
    mode: one of 'mse','kl','js'
    """
    if isinstance(scores, tuple):
        compute_device = scores[0].device
        concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in scores], dim=0)

    routing_weights = F.softmax(concatenated_gate_logits,dim=-1)
    B, M = routing_weights.shape

    I = routing_weights.mean(dim=0)  

    U = routing_weights.new_full((M,), 1.0 / M)

    if mode == "mse":

        loss = ((I - U) ** 2).mean()

    elif mode == "kl":
       
        loss = (I * (I.clamp(min=1e-8).log() + math.log(M))).sum()

    elif mode == "js":
        
        kl1 = (I * (I.clamp(min=1e-8).log() - U.clamp(min=1e-8).log())).sum()
        kl2 = (U * (U.clamp(min=1e-8).log() - I.clamp(min=1e-8).log())).sum()
        loss = 0.5 * (kl1 + kl2)

    else:
        raise ValueError(f"Unknown mode {mode}")

    return loss * num_experts

def load_balancing_loss_func(
    gate_logits: Union[torch.Tensor, Tuple[torch.Tensor], None],
    num_experts: Optional[int] = None,
    top_k=2,
    attention_mask= None,
) -> Union[torch.Tensor, int]:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits:
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        num_experts:
            Number of experts
        top_k:
            The number of experts to route per-token, can be also interpreted as the `top-k` routing
            parameter.
        attention_mask (`torch.Tensor`, *optional*):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)
    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)


    if attention_mask is None:
        
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )

        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
            expert_attention_mask, dim=0
        )

        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
            router_per_expert_attention_mask, dim=0
        )

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


class Qwen3MoeForCausalLM_LPR(Qwen3MoeForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3MoeModel_LPR(config) # Modification 18: use modified Model
        self.router_kl_loss_coef = config.router_kl_loss_coef # Modification 19: lpr regulation loss
        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        output_router_kl: Optional[bool] = None,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> MoeCausalLMOutputWithPast_KL:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_router_kl = (output_router_kl if output_router_kl is not None else self.config.output_router_kl) # Modification 20: lpr regulation loss
        outputs: MoeModelOutputWithPast_KL = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            cache_position=cache_position,
            output_router_kl = output_router_kl,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        
        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)
        net_loss = loss.item()
        aux_loss = 0.
        kl_loss = 0. 
        if output_router_logits:
         
            aux_loss = load_balancing_loss_func(
                gate_logits = outputs.router_logits,
                num_experts = self.num_experts,
                top_k = self.num_experts_per_tok,
                attention_mask =attention_mask,
            )
            aux_loss = self.router_aux_loss_coef * aux_loss

        if output_router_kl: # Modification 20: calculate regulation loss
            kl_loss = sum(outputs.kl_losses)
            kl_loss = kl_loss * self.router_kl_loss_coef 
            # loss += kl_loss.to(loss.device) *self.router_kl_loss_coef 
        if self.training:
            if labels is not None:
                loss +=  aux_loss.to(loss.device) 
                loss += kl_loss.to(loss.device) 

        return MoeCausalLMOutputWithPast_KL(  # Modification 21: calculate regulation loss
            loss=loss,
            net_loss=net_loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
            kl_loss=kl_loss ,

        )
