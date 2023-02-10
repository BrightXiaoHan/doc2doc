from typing import Dict, List, Optional

import torch
import torch.nn as nn
from fairseq.distributed import fsdp_wrap
from fairseq.models.transformer import TransformerEncoderBase
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.transformer_layer import TransformerEncoderLayerBase
from torch import Tensor

from .utils import Linear


class GTransformerEncoderLayer(TransformerEncoderLayerBase):
    def __init__(self, cfg, return_fc=False, global_ctx=False):
        super().__init__(cfg, return_fc)
        self.global_ctx = global_ctx
        if global_ctx:
            self.self_attn_local = self.build_self_attention(self.embed_dim, cfg)
            self.self_attn_gate = nn.Sequential(Linear(self.embed_dim * 2, self.embed_dim), nn.Sigmoid())
        else:
            self.self_attn_local = None
            self.self_attn_gate = None

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        if not self.global_ctx:
            return
        attn_name_map = {"self_attn": "self_attn_local"}
        for old, new in attn_name_map.items():
            for m in [
                "in_proj_weight",
                "in_proj_bias",
                "out_proj.weight",
                "out_proj.bias",
                "k_proj.weight",
                "k_proj.bias",
                "q_proj.weight",
                "q_proj.bias",
                "v_proj.weight",
                "v_proj.bias",
            ]:
                k_old = "{}.{}.{}".format(name, old, m)
                k_new = "{}.{}.{}".format(name, new, m)
                if k_old in state_dict and k_new not in state_dict:
                    state_dict[k_new] = state_dict[k_old].clone()

        # if initailized from a normal transformer, we need to add the attn gate
        attn_gate_weight_name = "{}.self_attn_gate.0.weight".format(name)
        attn_gate_bias_name = "{}.self_attn_gate.0.bias".format(name)

        if attn_gate_weight_name not in state_dict:
            # init with the original value
            state_dict[attn_gate_weight_name] = self.self_attn_gate[0].weight.clone()

        if attn_gate_bias_name not in state_dict:
            state_dict[attn_gate_bias_name] = self.self_attn_gate[0].bias.clone()

    def forward(
        self,
        x,
        encoder_padding_mask: Optional[Tensor],
        attn_mask: Optional[Tensor] = None,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8 if x.dtype == torch.float32 else -1e4)

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # Key changes of GTransformer
        if self.global_ctx:
            x_local, _ = self.self_attn_local(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask,
                need_weights=False,
                attn_mask=attn_mask,
            )
            x_global, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask, need_weights=False)
            gate = self.self_attn_gate(torch.cat([x_local, x_global], dim=-1))
            x = gate * x_local + (1 - gate) * x_global
        else:
            x, _ = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask,
                need_weights=False,
                attn_mask=attn_mask,
            )
        # Key changes of GTransformer end

        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)

        fc_result = x

        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)

        if self.return_fc and not torch.jit.is_scripting():
            return x, fc_result
        return x


class GTransformerEncoder(TransformerEncoderBase):
    def __init__(self, cfg, dictionary, embed_tokens, return_fc=False):
        origin_encoder_layers = cfg.encoder_layers
        gtransformer_encoder_ctx_layers = cfg.encoder_ctx_layers
        assert origin_encoder_layers > gtransformer_encoder_ctx_layers
        cfg.encoder_layers = origin_encoder_layers - gtransformer_encoder_ctx_layers
        super().__init__(cfg, dictionary, embed_tokens, return_fc)
        cfg.encoder_layers = origin_encoder_layers
        for _ in range(gtransformer_encoder_ctx_layers):
            self.layers.extend([self.build_group_encoder_layer(cfg)])

    @staticmethod
    def tokens2tags(dict, tokens):
        """
        generate group-tags according token sequence

        Given a token sequence "Hi . </s> I am a student . </s> I like NLP . <pad> <pad> <pad>"
        The tags should be "1 1 1 2 2 2 2 2 2 3 3 3 3 3 0 0 0"
        """

        def _toks2tags(tokens):
            tags = []
            next_tag = 1
            for tok in tokens:
                if tok in [dict.pad_index]:
                    tags.append(0)
                else:
                    tags.append(next_tag)
                    if tok == dict.eos_index:  # increase tag per </s>
                        next_tag += 1
            return tags

        tok_tags = [_toks2tags(tokens) for tokens in tokens.data.cpu().numpy().tolist()]
        tok_tags = torch.tensor(tok_tags, dtype=tokens.dtype, device=tokens.device)
        return tok_tags

    def build_encoder_layer(self, cfg):
        layer = GTransformerEncoderLayer(cfg, return_fc=self.return_fc)
        # migarate from super class
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def build_group_encoder_layer(self, cfg):
        layer = GTransformerEncoderLayer(cfg, return_fc=self.return_fc)

        # migarate from super class
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def get_local_attn_mask(self, src_tags):
        # Here we generate attention mask for Group Attention according to group tags
        local_attn_mask = src_tags.unsqueeze(1) != src_tags.unsqueeze(2)
        local_attn_mask &= 0 != src_tags.unsqueeze(2)

        # broadcast from shape (batch, seq_len, dim) to (batch * num_heads, seq_len, dim)
        local_attn_mask = local_attn_mask.unsqueeze(1).repeat(1, self.layers[0].self_attn.num_heads, 1, 1)
        local_attn_mask = local_attn_mask.view(-1, local_attn_mask.size(2), local_attn_mask.size(3))
        return local_attn_mask

    def forward_scriptable(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = torch.tensor(src_tokens.device.type == "xla") or encoder_padding_mask.any()
        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

        # account for padding while computing the representation
        x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x) * has_pads.type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []
        fc_results = []

        if return_all_hiddens:
            encoder_states.append(x)

        src_tags = self.tokens2tags(self.dictionary, src_tokens)
        local_attn_mask = self.get_local_attn_mask(src_tags)

        # encoder layers
        for layer in self.layers:
            if isinstance(layer, GTransformerEncoderLayer):
                attn_mask = local_attn_mask
            else:
                attn_mask = None
            lr = layer(x, encoder_padding_mask=encoder_padding_mask if has_pads else None, attn_mask=attn_mask)

            if isinstance(lr, tuple) and len(lr) == 2:
                x, fc_result = lr
            else:
                x = lr
                fc_result = None

            if return_all_hiddens and not torch.jit.is_scripting():
                assert encoder_states is not None
                encoder_states.append(x)
                fc_results.append(fc_result)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        src_lengths = src_tokens.ne(self.padding_idx).sum(dim=1, dtype=torch.int32).reshape(-1, 1).contiguous()
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "encoder_tags": src_tags,
            "fc_results": fc_results,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [src_lengths],
        }

    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        new_encoder_out = super().reorder_encoder_out(encoder_out, new_order)
        if len(encoder_out["encoder_tags"]) == 0:
            encoder_tags = []
        else:
            encoder_tags = encoder_out["encoder_tags"].index_select(0, new_order)
        new_encoder_out["encoder_tags"] = encoder_tags
        return new_encoder_out
