import jax
import jax.numpy as jnp

class Gemma3Config:
    """
    Config specifically for Gemma 3 12B inference extracting hidden states.
    Uses __getattr__ to dynamically mock any missing properties expected by MaxText.
    """
    def __init__(self):
        self.model_name = "gemma3-12b"
        
        # 12B specific parameters (from maxtext config)
        self.vocab_size = 262144
        self.emb_dim = 3840
        self.num_hidden_layers = 48
        self.num_query_heads = 16
        self.num_kv_heads = 8
        self.head_dim = 256
        self.mlp_dim = 15360
        
        # RoPE / Attention
        self.max_target_length = 8192
        self.max_prefill_predict_length = 8192
        self.sliding_window_size = 1024
        self.attn_logits_soft_cap = 50.0
        self.local_rope_max_timescale = 10000
        self.rope_max_timescale = 1000000
        self.rope_min_timescale = 10000
        self.rope_linear_scaling_factor = 8.0
        self.rope_type = "default"
        self.rope_use_scale = False
        self.normalization_layer_epsilon = 1e-6
        self.decoder_block = "gemma3"
        self.attention = "dot_product"
        
        # Internal MaxText explicit overrides
        self.attention_type = "local_sliding"
        
        # Normalizations
        self.use_post_attn_norm = True
        self.use_post_ffw_norm = True
        self.use_qk_norm = True
        
        # Execution modes
        self.dtype = jnp.bfloat16
        self.weight_dtype = jnp.bfloat16
        self.float32_qk_product = True
        self.float32_logits = True
        
        # Features we don't need for inference
        self.dropout_rate = 0.0
        self.record_internal_nn_metrics = False
        self.scan_layers = False

    def __getattr__(self, name):
        """
        Dynamically catches any missing configuration attributes expected by MaxText's underlying layers.
        Provides safe default fallbacks without needing to explicitly define hundreds of flags.
        """
        # Return generic safe defaults based on naming conventions or general MaxText defaults
        if "size" in name or "length" in name or "timescale" in name or "parallelism" in name:
            return 1 if "parallelism" in name else -1
        elif "enable" in name or "use" in name or "is_" in name or "record" in name:
            return False
        elif "dtype" in name or "precision" in name:
            return "default"
        elif "quant" in name or "name" in name or "policy" in name or "backend" in name or "mode" in name:
            return "default" if name == "shard_mode" else ""
        elif "factor" in name or "scaling" in name or "cap" in name or "beta" in name:
            return 1.0
        elif "rules" in name or "axes" in name or "axis" in name:
            return ()
        elif "activations" in name:
            return ["gelu", "linear"]
        
        # Absolute catch-all
        return None
