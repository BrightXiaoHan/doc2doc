from typing import Dict, List, Optional

import torch
import torch.nn as nn
from fairseq.models.transformer import TransformerDecoderBase
from fairseq.modules.transformer_layer import TransformerDecoderLayerBase
from torch import Tensor

from .utils import Linear


class GTransformerDecoderLayer(TransformerDecoderLayerBase):
    def __init__(self, cfg, global_ctx=False, **kwargs):
        super().__init__(cfg, **kwargs)

        self.global_ctx = global_ctx
        if global_ctx:
            self.self_attn_local = self.build_self_attention(self.embed_dim, cfg)
            self.self_attn_gate = nn.Sequential(Linear(self.embed_dim * 2, self.embed_dim), nn.Sigmoid())
            self.encoder_attn_local = self.build_encoder_attention(self.embed_dim, cfg)
            self.encoder_attn_gate = nn.Sequential(Linear(self.embed_dim * 2, self.embed_dim), nn.Sigmoid())

    def upgrade_state_dict_named(self, state_dict, name):
        if not self.global_ctx:
            return
        attn_name_map = {
            "self_attn": "self_attn_local",
            "encoder_attn": "encoder_attn_local",
        }
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
        for gate_name in ["self_attn_gate", "encoder_attn_gate"]:
            attn_gate_weight_name = "{}.{}.0.weight".format(name, gate_name)
            attn_gate_bias_name = "{}.{}.0.bias".format(name, gate_name)

            if attn_gate_weight_name not in state_dict:
                # init with the original value
                state_dict[attn_gate_weight_name] = self.self_attn_gate[0].weight.clone()

            if attn_gate_bias_name not in state_dict:
                state_dict[attn_gate_bias_name] = self.self_attn_gate[0].bias.clone()

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
        local_encoder_attn_mask: Optional[torch.Tensor] = None,
        local_self_attn_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)

        if self.cross_self_attention and not (
            incremental_state is not None and _self_attn_input_buffer is not None and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat((x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1)
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(encoder_out.size(1), encoder_out.size(0))
                self_attn_padding_mask = torch.cat((encoder_padding_mask, self_attn_padding_mask), dim=1)
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        if self.global_ctx:
            x_global, _ = self.self_attn(
                query=x,
                key=y,
                value=y,
                key_padding_mask=self_attn_padding_mask,
                incremental_state=incremental_state,
                need_weights=False,
                attn_mask=self_attn_mask,
            )

            x_local, _ = self.self_attn_local(
                query=x,
                key=y,
                value=y,
                key_padding_mask=self_attn_padding_mask,
                incremental_state=incremental_state,
                need_weights=False,
                attn_mask=local_self_attn_mask,
            )
            gate = self.self_attn_gate(torch.cat([x_local, x_global], dim=-1))
            x = gate * x_local + (1 - gate) * x_global
        else:
            x, _ = self.self_attn(
                query=x,
                key=y,
                value=y,
                key_padding_mask=self_attn_padding_mask,
                incremental_state=incremental_state,
                need_weights=False,
                attn_mask=local_self_attn_mask,
            )

        if self.c_attn is not None:
            tgt_len, bsz = x.size(0), x.size(1)
            x = x.view(tgt_len, bsz, self.nh, self.head_dim)
            x = torch.einsum("tbhd,h->tbhd", x, self.c_attn)
            x = x.reshape(tgt_len, bsz, self.embed_dim)
        if self.attn_ln is not None:
            x = self.attn_ln(x)

        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            if self.global_ctx:
                x_local, _ = self.encoder_attn_local(
                    query=x,
                    key=encoder_out,
                    value=encoder_out,
                    key_padding_mask=encoder_padding_mask,
                    incremental_state=incremental_state,
                    static_kv=True,
                    need_weights=need_attn or (not self.training and self.need_attn),
                    need_head_weights=need_head_weights,
                    attn_mask=local_encoder_attn_mask,
                )
                x_global, attn = self.encoder_attn(
                    query=x,
                    key=encoder_out,
                    value=encoder_out,
                    key_padding_mask=encoder_padding_mask,
                    incremental_state=incremental_state,
                    static_kv=True,
                    need_weights=need_attn or (not self.training and self.need_attn),
                    need_head_weights=need_head_weights,
                )
                gate = self.encoder_attn_gate(torch.cat([x_local, x_global], dim=-1))
                x = gate * x_local + (1 - gate) * x_global

            else:
                x, attn = self.encoder_attn(
                    query=x,
                    key=encoder_out,
                    value=encoder_out,
                    key_padding_mask=encoder_padding_mask,
                    incremental_state=incremental_state,
                    static_kv=True,
                    need_weights=need_attn or (not self.training and self.need_attn),
                    need_head_weights=need_head_weights,
                    attn_mask=local_encoder_attn_mask,
                )

            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        if self.w_resid is not None:
            residual = torch.mul(self.w_resid, residual)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None


class GTransformerDecoder(TransformerDecoderBase):
    def __init__(self, cfg, *args, **kwargs):
        origin_decoder_layers = cfg.decoder_layers
        gtransformer_decoder_ctx_layers = cfg.decoder_ctx_layers
        assert gtransformer_decoder_ctx_layers <= origin_decoder_layers
        cfg.decoder_layers = origin_decoder_layers - gtransformer_decoder_ctx_layers
        super().__init__(cfg, *args, **kwargs)
        cfg.decoder_layers = origin_decoder_layers
        for _ in range(cfg.decoder_ctx_layers):
            self.layers.extend([self.build_global_decoder_layer(cfg)])

    @staticmethod
    def tokens2tags(dict, tokens):
        """
        generate group-tags according token sequence

        Suppose we have a token seq "</s> I am a student </s> You are a student <pad>", then the group-tags
        [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 0]
        """

        def _toks2tags(tokens):
            next_tag = 1
            tags = [next_tag]
            for tok in tokens[1:]:
                if tok in [dict.pad_index]:
                    tags.append(0)
                else:
                    if tok == dict.eos_index:  # increase tag per </s>
                        next_tag += 1
                    tags.append(next_tag)
            return tags

        tok_tags = [_toks2tags(tokens) for tokens in tokens.data.cpu().numpy().tolist()]
        tok_tags = torch.tensor(tok_tags, dtype=tokens.dtype, device=tokens.device)
        return tok_tags

    def build_decoder_layer(self, cfg, no_encoder_attn=False):
        layer = GTransformerDecoderLayer(cfg, no_encoder_attn=no_encoder_attn, global_ctx=False)
        return layer

    def build_global_decoder_layer(self, cfg, no_encoder_attn=False):
        layer = GTransformerDecoderLayer(cfg, no_encoder_attn=no_encoder_attn, global_ctx=True)
        return layer

    def get_local_encoder_attn_mask(self, encoder_tags, decoder_tags):
        # Handle beamsize
        beamsize = decoder_tags.size(0) // encoder_tags.size(0)
        if beamsize > 1:
            encoder_tags = encoder_tags.repeat(beamsize, 1).view(-1, encoder_tags.size(-1))
        # G-Transformer local attention mask for cross-attention
        attn_mask = encoder_tags.unsqueeze(1) != decoder_tags.unsqueeze(2)
        attn_mask &= 0 != decoder_tags.unsqueeze(2)

        # broadcast from shape (batch, seq_len, dim) to (batch * num_heads, seq_len, dim)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.layers[0].self_attn.num_heads, 1, 1)
        attn_mask = attn_mask.view(-1, attn_mask.size(2), attn_mask.size(3))
        return attn_mask

    def get_local_self_attn_mask(self, decoder_tags):
        # G-Transformer local attention mask for self-attention
        attn_mask = decoder_tags.unsqueeze(1) != decoder_tags.unsqueeze(2)
        attn_mask &= 0 != decoder_tags.unsqueeze(2)

        # broadcast from shape (batch, seq_len, dim) to (batch * num_heads, seq_len, dim)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.layers[0].self_attn.num_heads, 1, 1)
        attn_mask = attn_mask.view(-1, attn_mask.size(2), attn_mask.size(3))
        triu_mask = torch.triu(torch.ones_like(attn_mask[0]), diagonal=1)
        attn_mask = attn_mask | triu_mask
        return attn_mask

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            enc = encoder_out["encoder_out"][0]
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            padding_mask = encoder_out["encoder_padding_mask"][0]

        # embed positions
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(prev_output_tokens, incremental_state=incremental_state)

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # Prevent torchscript exporting issue for dynamic quant embedding
        prev_output_tokens = prev_output_tokens.contiguous()
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # G-Transformer embed tags
        decoder_tags = self.tokens2tags(self.dictionary, prev_output_tokens)
        local_encoder_attn_mask = self.get_local_encoder_attn_mask(encoder_out["encoder_tags"], decoder_tags)
        local_self_attn_mask = self.get_local_self_attn_mask(decoder_tags)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            if isinstance(layer, GTransformerDecoderLayer):
                x, layer_attn, _ = layer(
                    x,
                    enc,
                    padding_mask,
                    incremental_state,
                    self_attn_mask=self_attn_mask,
                    self_attn_padding_mask=self_attn_padding_mask,
                    need_attn=bool((idx == alignment_layer)),
                    need_head_weights=bool((idx == alignment_layer)),
                    local_encoder_attn_mask=local_encoder_attn_mask,
                    local_self_attn_mask=local_self_attn_mask,
                )
            else:
                x, layer_attn, _ = layer(
                    x,
                    enc,
                    padding_mask,
                    incremental_state,
                    self_attn_mask=self_attn_mask,
                    self_attn_padding_mask=self_attn_padding_mask,
                    need_attn=bool((idx == alignment_layer)),
                    need_head_weights=bool((idx == alignment_layer)),
                )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}
