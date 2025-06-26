import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import  List, Optional, Tuple, Union
from functools import partial
from transformers.cache_utils import Cache
from transformers.cache_utils import DynamicCache
from transformers.processing_utils import Unpack
from transformers.utils import logging, LossKwargs, can_return_tuple
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from .router import TokenDistributionRouter
from transformers.modeling_outputs import ( MoeCausalLMOutputWithPast,
                                           BaseModelOutputWithPast,
                                           CausalLMOutputWithPast,
                                            MoeModelOutputWithPast)
from transformers import (
    DeepseekV3Config,
    DeepseekV3Model,
    DeepseekV3ForCausalLM,
    DeepseekV3MoE,
    DeepseekV3DecoderLayer,
)
from dataclasses import dataclass
logger = logging.get_logger(__name__)

class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...

@dataclass
class BaseModelOutputWithPast_KL(BaseModelOutputWithPast):
    kl_losses: Optional[Tuple[torch.FloatTensor]] = None
    router_logits: Optional[Tuple[torch.FloatTensor]] = None
@dataclass
class CausalLMOutputWithPast_KL(CausalLMOutputWithPast):
    kl_loss: Optional[Tuple[torch.FloatTensor]] = None
    net_loss: Optional[torch.FloatTensor] = None
    router_logits: Optional[Tuple[torch.FloatTensor]] = None
class DeepseekV3Config_LPR(DeepseekV3Config):
    model_type = "deepseekv3_lpr"
    def __init__(
        self,
        router_kl_loss_coef=0.01,
        output_router_kl=False,
        output_router_logit = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.output_router_kl = output_router_kl ###
        self.output_router_logits = output_router_logit
        self.router_kl_loss_coef = router_kl_loss_coef
class DeepseekV3MoE_LPR(DeepseekV3MoE):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config,layer_idx):
        super().__init__(config)
        self.gate = TokenDistributionRouter(config,layer_idx)
    def moe(self, hidden_states: torch.Tensor, topk_indices: torch.Tensor, topk_weights: torch.Tensor):
        r"""
        CALL FOR CONTRIBUTION! I don't have time to optimise this right now, but expert weights need to be fused
        to not have to do a loop here (deepseek has 256 experts soooo yeah).
        """
        final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
        expert_mask = torch.nn.functional.one_hot(topk_indices, num_classes=len(self.experts))
        expert_mask = expert_mask.permute(2, 0, 1)

        for expert_idx in range(len(self.experts)):
            expert = self.experts[expert_idx]
            mask = expert_mask[expert_idx]
            token_indices, weight_indices = torch.where(mask)

            if token_indices.numel() > 0:
                expert_weights = topk_weights[token_indices, weight_indices]
                expert_input = hidden_states[token_indices]
                expert_output = expert(expert_input)
                weighted_output = expert_output * expert_weights.unsqueeze(-1)
                final_hidden_states.index_add_(0, token_indices, weighted_output)

        # in original deepseek, the output of the experts are gathered once we leave this module
        # thus the moe module is itelsf an IsolatedParallel module
        # and all expert are "local" meaning we shard but we don't gather
        return final_hidden_states.type(hidden_states.dtype)

    def forward(self, hidden_states):
        residuals = hidden_states
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        # topk_indices, topk_weights = self.gate(hidden_states)
        topk_weights, kl, topk_indices, logits, z_decoed = self.gate(hidden_states)

        hidden_states = hidden_states+ z_decoed
        hidden_states = self.moe(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states, kl, logits



class DeepseekV3DecoderLayer_LPR(DeepseekV3DecoderLayer):
    def __init__(self, config: DeepseekV3Config, layer_idx: int):
        super().__init__(config, layer_idx)
        if layer_idx >= config.first_k_dense_replace:
            self.mlp = DeepseekV3MoE_LPR(config,layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        output_router_kl: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
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
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states, kl, logits = hidden_states
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if output_router_kl:
            outputs += (kl,)
        if output_router_logits:
            outputs += (logits,)
        return outputs

class DeepseekV3Model_LPR(DeepseekV3Model):
    def __init__(self, config: DeepseekV3Config_LPR):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [DeepseekV3DecoderLayer_LPR(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_router_kl: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast_KL:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

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

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_kl = () if output_router_kl else None
        all_router_logits = () if output_router_logits else None
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
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
                    use_cache,
                    cache_position,
                    position_embeddings,
                    output_router_kl,
                    output_router_logits

                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    output_router_kl = output_router_kl,
                    output_router_logits = output_router_logits,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            if output_router_logits:
                all_router_logits += (layer_outputs[-1],)
            if output_router_kl:
                all_router_kl += (layer_outputs[1],)
                
        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast_KL(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            kl_losses= all_router_kl,
            router_logits = all_router_logits
        )

class DeepseekV3ForCausalLM_LPR(DeepseekV3ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = DeepseekV3Model_LPR(config)
        self.router_kl_loss_coef = config.router_kl_loss_coef
        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        output_router_kl: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast_KL:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_router_kl = (output_router_kl if output_router_kl is not None else self.config.output_router_kl)
        output_router_logits = (output_router_logits if output_router_logits is not None else self.config.output_router_logits)
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast_KL = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            output_router_kl = output_router_kl,
            output_router_logits = output_router_logits,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
        net_loss = loss.item()

        kl_loss = 0.       # make sure to reside in the same device
        if output_router_kl:
            kl_loss = sum(outputs.kl_losses)
            loss += kl_loss.to(loss.device) *self.router_kl_loss_coef 

        return CausalLMOutputWithPast_KL(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            kl_loss=kl_loss ,
            router_logits = outputs.router_logits,
            net_loss=net_loss,
        )