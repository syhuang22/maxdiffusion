import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from transformers import AutoTokenizer

from .config import Gemma3Config
from .gemma3_model import Gemma3TextEncoderModel

class Gemma3TextEncoder:
    """
    Wrapper for Gemma 3 to act as a text encoder providing all hidden states.
    This acts as the entry point for the Feature Extractor.
    """
    def __init__(self, model_name_or_path: str = "google/gemma-3-12b-it"):
        self.config = Gemma3Config()
        # Initialize the JAX/Flax model architecture
        self.model = Gemma3TextEncoderModel(config=self.config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def encode(self, text: str, params: FrozenDict):
        """
        Encodes text into hidden states across all layers.
        
        Args:
            text: A single string or list of strings to encode.
            params: The loaded JAX/Flax weights for Gemma 3.
            
        Returns:
            A tuple of jax Arrays representing the hidden states from the 
            embedding layer up to the final transformer layer.
            Shape of each element: (batch_size, seq_len, hidden_dim)
        """
        # Tokenize the input text
        inputs = self.tokenizer(
            text, 
            return_tensors="np", 
            padding="max_length", 
            max_length=self.config.max_target_length,
            truncation=True
        )
        
        input_ids = jnp.array(inputs['input_ids'])
        
        # Calculate positions
        # Simple position IDs calculation: 0 to sequence_length - 1
        batch_size, seq_len = input_ids.shape
        decoder_positions = jnp.broadcast_to(jnp.arange(seq_len), (batch_size, seq_len))
        
        # Forward pass through the model to get all hidden states
        # The model returns a tuple of length `num_hidden_layers + 1` 
        # (embeddings + 48 layers for 12B model)
        all_hidden_states = self.model.apply(
            {'params': params}, 
            input_ids=input_ids,
            decoder_positions=decoder_positions,
            deterministic=True
        )
        
        return all_hidden_states
