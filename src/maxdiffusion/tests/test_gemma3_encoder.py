import sys
import os
import pytest
import jax
import jax.numpy as jnp
from transformers import AutoTokenizer

# Dynamically add maxtext to sys.path so we can run the test locally
MAXTEXT_PATH = "/home/shyhuang_google_com/maxtext"
if os.path.exists(MAXTEXT_PATH):
    sys.path.insert(0, os.path.join(MAXTEXT_PATH, "src"))

MAXDIFFUSION_PATH = "/home/shyhuang_google_com/maxdiffusion/src"
if os.path.exists(MAXDIFFUSION_PATH):
    sys.path.insert(0, MAXDIFFUSION_PATH)

from maxdiffusion.models.gemma3.gemma3_encoder import Gemma3TextEncoder


class TestGemma3TextEncoder:
    """Test suite for the Gemma 3 Text Encoder."""
    
    @pytest.fixture(scope="class")
    def encoder(self):
        """Fixture to initialize the encoder once for all tests in this class."""
        print("Initializing Gemma3TextEncoder...")
        encoder = Gemma3TextEncoder("google/gemma-3-12b-it")
        # For testing purposes, limit sequence length to avoid OOM on smaller machines
        encoder.config.max_target_length = 32
        encoder.config.max_prefill_predict_length = 16
        encoder.config.sliding_window_size = 16
        return encoder

    def test_encoder_initialization(self, encoder):
        """Verify that the model config is loaded correctly."""
        assert encoder.config.num_hidden_layers == 48
        assert encoder.config.emb_dim == 3840
        assert encoder.config.max_target_length == 32
        
    def test_forward_pass_shapes(self, encoder):
        """Verify that the forward pass returns the expected number of layers and shapes."""
        batch_size = 1
        seq_len = 16
        
        # Set up a distributed Mesh across all available TPUs (e.g. 8 devices for v6e-8)
        devices = jax.devices()
        print(f"   Found {len(devices)} JAX devices. Setting up Mesh...")
        
        # 1. Initialize dummy parameters in a distributed, sharded way to prevent OOM
        # By JIT compiling the init function and placing the output on the mesh,
        # JAX will automatically shard the 24GB weights across all 8 devices.
        dummy_input_ids = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
        dummy_positions = jnp.broadcast_to(jnp.arange(seq_len), (batch_size, seq_len))
        
        mesh = jax.sharding.Mesh(devices, ('data',))
        
        # We specify that the inputs are replicated (not sharded) but the parameters 
        # inside the model will be sharded automatically by Flax if configured, or 
        # simply distributed by XLA to fit memory.
        @jax.jit
        def init_fn():
            return encoder.model.init(
                jax.random.PRNGKey(0), 
                input_ids=dummy_input_ids,
                decoder_positions=dummy_positions,
                deterministic=True
            )
            
        with mesh:
            variables = init_fn()
            params = variables['params']
            print("   Dummy parameters initialized and sharded successfully!")
            
            # 2. Run the actual encode function within the mesh context
            prompt = "A test prompt for Gemma3"
            
            # Tokenize inside the test to bypass the numpy to jax array transition outside mesh
            inputs = encoder.tokenizer(
                prompt, 
                return_tensors="np", 
                padding="max_length", 
                max_length=encoder.config.max_target_length,
                truncation=True
            )
            input_ids_jnp = jnp.array(inputs['input_ids'])
            dec_pos = jnp.broadcast_to(jnp.arange(input_ids_jnp.shape[1]), (1, input_ids_jnp.shape[1]))
            
            @jax.jit
            def forward_fn(p, ids, pos):
                return encoder.model.apply(
                    {'params': p}, 
                    input_ids=ids,
                    decoder_positions=pos,
                    deterministic=True
                )
                
            print("   Compiling and running forward pass...")
            all_hidden_states = forward_fn(params, input_ids_jnp, dec_pos)
        
        # 3. Assertions and Integration with Feature Extractor
        expected_layers = encoder.config.num_hidden_layers + 1 # 48 + 1 = 49
        assert len(all_hidden_states) == expected_layers, f"Expected {expected_layers} layers, got {len(all_hidden_states)}"
        
        # Validate that we can concatenate all 49 layers for the LTX-2 Feature Extractor
        # The PyTorch feature extractor expects shape (batch_size, 3840 * 49)
        # However, because we handle sequences, it will be (batch_size, seq_len, 3840 * 49) before linear projection
        
        # Stack shape: (49, batch, seq_len, emb_dim)
        stacked_states = jnp.stack(all_hidden_states, axis=0)
        
        # Transpose to: (batch, seq_len, 49, emb_dim)
        transposed_states = jnp.transpose(stacked_states, (1, 2, 0, 3))
        
        # Reshape (flatten the last two dimensions) to: (batch, seq_len, 49 * emb_dim)
        flattened_feature = jnp.reshape(
            transposed_states, 
            (batch_size, encoder.config.max_target_length, expected_layers * encoder.config.emb_dim)
        )
        
        expected_flat_shape = (batch_size, encoder.config.max_target_length, 49 * 3840)
        print(f"   Flattened feature shape for FeatureExtractor: {flattened_feature.shape}")
        assert flattened_feature.shape == expected_flat_shape, f"Expected {expected_flat_shape}, got {flattened_feature.shape}"
        
        # 4. Simulate the GemmaFeaturesExtractorProjLinear from LTX-2
        # self.aggregate_embed = torch.nn.Linear(3840 * 49, 3840, bias=False)
        dummy_rng = jax.random.PRNGKey(42)
        dummy_linear_weights = jax.random.normal(dummy_rng, (49 * 3840, 3840))
        final_embedding = jnp.dot(flattened_feature, dummy_linear_weights)
        
        expected_final_shape = (batch_size, encoder.config.max_target_length, 3840)
        print(f"   Final projected embedding shape: {final_embedding.shape}")
        assert final_embedding.shape == expected_final_shape, f"Expected {expected_final_shape}, got {final_embedding.shape}"
        
        print("\n🎉 Output perfectly matches the GemmaFeaturesExtractorProjLinear format!")
