import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import nnx
from typing import Optional, Any

from MaxText.common_types import AttentionType, MODEL_MODE_PREFILL
from maxtext.layers import initializers
from maxtext.layers.attentions import Attention
from maxtext.layers.linears import MlpBlock
from maxtext.layers.normalizations import RMSNorm
from maxtext.utils import max_utils
from maxtext.layers import nnx_wrappers

from .config import Gemma3Config

GEMMA3_ATTENTION_PATTERN = (
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.GLOBAL,
)

def get_attention_type(layer_id):
    layer_id %= len(GEMMA3_ATTENTION_PATTERN)
    return GEMMA3_ATTENTION_PATTERN[layer_id]

class Gemma3DecoderLayer(nnx.Module):
    """Transformer decoder layer for Gemma3 (NNX version)."""
    
    def __init__(
        self,
        config: Gemma3Config,
        rngs: nnx.Rngs,
        mesh: jax.sharding.Mesh,
        attention_type: AttentionType = AttentionType.LOCAL_SLIDING,
    ):
        self.config = config
        self.attention_type = attention_type
        
        cfg = config
        
        self.pre_self_attention_norm = RMSNorm(
            num_features=cfg.emb_dim,
            dtype=cfg.dtype,
            weight_dtype=cfg.weight_dtype,
            kernel_axes=("norm",),
            rngs=rngs,
        )

        # We need to provide dummy shapes for initialization as required by MaxText Attention
        # Batch size and sequence length aren't strictly required for weight shapes, but we provide generics.
        dummy_inputs_shape = (1, cfg.max_target_length, cfg.emb_dim)
        
        query_pre_attn_scalar = cfg.head_dim**-0.5
        self.self_attention = Attention(
            config=cfg,
            num_query_heads=cfg.num_query_heads,
            num_kv_heads=cfg.num_kv_heads,
            head_dim=cfg.head_dim,
            max_target_length=cfg.max_target_length,
            max_prefill_predict_length=cfg.max_prefill_predict_length,
            attention_kernel="dot_product",
            inputs_q_shape=dummy_inputs_shape,
            inputs_kv_shape=dummy_inputs_shape,
            mesh=mesh,
            dtype=cfg.dtype,
            weight_dtype=cfg.weight_dtype,
            dropout_rate=cfg.dropout_rate,
            float32_qk_product=cfg.float32_qk_product,
            float32_logits=cfg.float32_logits,
            attention_type=self.attention_type,
            sliding_window_size=cfg.sliding_window_size,
            attn_logits_soft_cap=cfg.attn_logits_soft_cap,
            use_qk_norm=True,
            query_pre_attn_scalar=query_pre_attn_scalar,
            model_mode=MODEL_MODE_PREFILL,
            rngs=rngs,
        )

        if cfg.use_post_attn_norm:
            self.post_self_attention_norm = RMSNorm(
                num_features=cfg.emb_dim,
                dtype=cfg.dtype,
                weight_dtype=cfg.weight_dtype,
                kernel_axes=("norm",),
                rngs=rngs,
            )
        else:
            self.post_self_attention_norm = None

        self.pre_ffw_norm = RMSNorm(
            num_features=cfg.emb_dim,
            dtype=cfg.dtype,
            weight_dtype=cfg.weight_dtype,
            kernel_axes=("norm",),
            rngs=rngs,
        )

        self.mlp = MlpBlock(
            in_features=cfg.emb_dim,
            intermediate_dim=cfg.mlp_dim,
            activations=["gelu", "linear"], 
            intermediate_dropout_rate=cfg.dropout_rate,
            dtype=cfg.dtype,
            weight_dtype=cfg.weight_dtype,
            config=cfg,
            mesh=mesh,
            model_mode=MODEL_MODE_PREFILL,
            rngs=rngs,
        )

        if cfg.use_post_ffw_norm:
            self.post_ffw_norm = RMSNorm(
                num_features=cfg.emb_dim,
                dtype=cfg.dtype,
                weight_dtype=cfg.weight_dtype,
                kernel_axes=("norm",),
                rngs=rngs,
            )
        else:
            self.post_ffw_norm = None

    def __call__(
        self,
        inputs,
        decoder_positions,
        decoder_segment_ids=None,
        deterministic=True,
    ):
        lnx = self.pre_self_attention_norm(inputs)
        
        attention_lnx, _ = self.self_attention(
            lnx,
            lnx,
            decoder_positions,
            decoder_segment_ids=decoder_segment_ids,
            deterministic=deterministic,
            model_mode=MODEL_MODE_PREFILL,
        )

        if self.post_self_attention_norm is not None:
            attention_lnx = self.post_self_attention_norm(attention_lnx)

        attention_lnx += inputs
        residual = attention_lnx

        attn_output = self.pre_ffw_norm(attention_lnx)

        mlp_lnx = self.mlp(attn_output, deterministic=deterministic)

        if self.post_ffw_norm is not None:
            mlp_lnx = self.post_ffw_norm(mlp_lnx)

        layer_output = mlp_lnx + residual

        return layer_output, layer_output

Gemma3DecoderLayerToLinen = nnx_wrappers.to_linen_class(
    Gemma3DecoderLayer,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)

class Gemma3BlockSequence(nnx.Module):
    """A sequence of Gemma3 decoder layers (NNX version) using a for loop."""
    def __init__(self, config: Gemma3Config, rngs: nnx.Rngs):
        self.config = config
        self.num_layers = config.num_hidden_layers
        
        # Create a basic 1D mesh for inference if none is provided
        # MaxText expects a mesh with at least some axes mapped to prevent 'NoneType' errors.
        devices = jax.devices()
        mesh = jax.sharding.Mesh(devices, ('tensor',))
        
        for layer_id in range(self.num_layers):
            attention_type = get_attention_type(layer_id)
            layer = Gemma3DecoderLayer(
                config=config,
                rngs=rngs,
                mesh=mesh,
                attention_type=attention_type,
            )
            setattr(self, f"layer_{layer_id}", layer)

    def __call__(
        self,
        inputs,
        decoder_positions,
        decoder_segment_ids=None,
        deterministic=True,
    ):
        x = inputs
        all_hidden_states = []

        for layer_id in range(self.num_layers):
            layer = getattr(self, f"layer_{layer_id}")
            x, _ = layer(
                x, 
                decoder_positions=decoder_positions,
                decoder_segment_ids=decoder_segment_ids,
                deterministic=deterministic
            )
            all_hidden_states.append(x)
            
        return jnp.stack(all_hidden_states, axis=0)

Gemma3BlockSequenceToLinen = nnx_wrappers.to_linen_class(
    Gemma3BlockSequence,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)

class Gemma3TextEncoderModel(nn.Module):
    """The main wrapper for Gemma 3 specifically for extracting text embeddings."""
    config: Gemma3Config

    @nn.compact
    def __call__(
        self,
        input_ids,
        decoder_positions,
        decoder_segment_ids=None,
        deterministic=True,
    ):
        cfg = self.config
        
        # Embedding Look-up
        # Gemma 3 multiplies embeddings by sqrt(hidden_dim)
        embeddings = nn.Embed(
            num_embeddings=cfg.vocab_size,
            features=cfg.emb_dim,
            dtype=cfg.dtype,
            name="token_embedder",
        )(input_ids)
        
        x = embeddings * jnp.sqrt(cfg.emb_dim)

        # Run through the transformer layers and get ALL hidden states
        # The shape of all_hidden_states will be (num_hidden_layers, batch_size, seq_len, emb_dim)
        all_hidden_states = Gemma3BlockSequenceToLinen(config=cfg, name="layers")(
            x,
            decoder_positions=decoder_positions,
            decoder_segment_ids=decoder_segment_ids,
            deterministic=deterministic
        )
        
        # If the feature extractor expects a list or tuple:
        # We can unstack the first dimension (layers)
        # We also usually want to include the initial embeddings at index 0
        hidden_states_list = [x] + [all_hidden_states[i] for i in range(cfg.num_hidden_layers)]
        
        return tuple(hidden_states_list)
