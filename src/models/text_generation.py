# src/models/text_generation.py
"""
MODULE: text_generation.py
PURPOSE: Provides utilities for text generation using language models.
KEY COMPONENTS:
- TextGenerator: Provides methods for auto-regressive text generation using various sampling strategies.
DEPENDENCIES: torch, torch.nn.functional, typing, numpy
SPECIAL NOTES: Supports temperature, top-k, and top-p sampling strategies.
"""

import os
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F


class TextGenerator:
    """
    Text generation utilities for language models.
    
    This class provides methods for auto-regressive text generation
    using various sampling strategies.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the text generator.
        
        Args:
            model: Language model for generation
            tokenizer: Tokenizer for encoding/decoding text
            device: Device to run on
        """
        self.model = model
        self.tokenizer = tokenizer

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else
                                     "mps" if torch.backends.mps.is_available() else
                                     "cpu")
        else:
            self.device = device

        # Move model to device
        self.model.to(self.device)

        # Put model in evaluation mode
        self.model.eval()

        # Get special token indices
        self.pad_idx = tokenizer.special_tokens["pad_token_idx"]
        self.bos_idx = tokenizer.special_tokens["bos_token_idx"]
        self.eos_idx = tokenizer.special_tokens["eos_token_idx"]

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        return_attention: bool = False,
    ) -> Union[List[str], Tuple[List[str], List[torch.Tensor]]]:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Text prompt to start generation
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Temperature for sampling (higher = more random)
            top_k: Number of highest probability tokens to consider (None = all)
            top_p: Cumulative probability threshold for nucleus sampling (None = disabled)
            do_sample: Whether to sample from the distribution (False = greedy)
            num_return_sequences: Number of sequences to generate
            return_attention: Whether to return attention patterns
            
        Returns:
            List of generated texts or
            Tuple of (List of generated texts, List of attention maps)
        """
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt)

        # Add BOS token if not present
        if len(input_ids) == 0 or input_ids[0] != self.bos_idx:
            input_ids = [self.bos_idx] + input_ids

        # Create tensor and move to device
        input_ids = torch.tensor([input_ids] * num_return_sequences, dtype=torch.long).to(self.device)

        # Track attention patterns if requested
        attention_maps: List[torch.Tensor] = []

        # Check if we're working with EncoderDecoderTransformer
        is_encoder_decoder = hasattr(self.model, 'encoder') and hasattr(self.model, 'decoder')

        # Generate auto-regressively
        for _ in range(max_new_tokens):
            # Create attention mask (all 1s for input tokens)
            attention_mask = torch.ones_like(input_ids)

            # Forward pass
            with torch.no_grad():
                # Set model attributes for attention tracking if needed
                if return_attention and hasattr(self.model, 'output_attentions'):
                    setattr(self.model, 'output_attentions', True)

                # Get outputs - handle different model interfaces
                if is_encoder_decoder:
                    # For encoder-decoder models like EncoderDecoderTransformer
                    # Use src for encoding and tgt for decoding
                    src = input_ids
                    tgt = input_ids[:, -1:] # Last token for next prediction
                    src_mask = attention_mask

                    # First encode the source
                    memory = self.model.encode(src, src_mask=src_mask)

                    # Then decode with memory for single token prediction
                    outputs = self.model.decode(tgt, memory)

                    # In this case, outputs should already be probabilities from softmax
                    logits = outputs
                else:
                    # For standard models that accept input_ids and attention_mask
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )

                    # Get logits
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs

                # Store attention maps if requested
                if return_attention and hasattr(outputs, 'attentions') and outputs.attentions is not None:
                    attention_maps.extend(outputs.attentions)

                # Focus on the last token prediction
                next_token_logits = logits[:, -1, :] if not is_encoder_decoder else logits.squeeze(1)

                # Adjust prediction with temperature
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature

                # Apply sampling methods
                if do_sample:
                    # Apply top-k filtering
                    if top_k is not None:
                        indices_to_remove = torch.topk(next_token_logits, top_k)[0][:, -1].unsqueeze(-1) <= next_token_logits
                        next_token_logits = next_token_logits.masked_fill(indices_to_remove, -float('Inf'))

                    # Apply top-p (nucleus) filtering
                    if top_p is not None:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs > top_p

                        # Shift indices to remove the first token (keep at least one token)
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0

                        # Scatter sorted indices to original indexing
                        indices_to_remove = sorted_indices_to_remove.scatter(
                            1, sorted_indices, sorted_indices_to_remove
                        )
                        next_token_logits = next_token_logits.masked_fill(indices_to_remove, -float('Inf'))

                    # Sample from the filtered distribution
                    if is_encoder_decoder:
                        # For encoder-decoder, we need to make sure we have valid probabilities
                        # Ensure no inf, -inf, or nan values
                        next_token_logits = torch.where(
                            torch.isfinite(next_token_logits),
                            next_token_logits,
                            torch.zeros_like(next_token_logits)
                        )
                        # Add small epsilon to avoid zeros
                        next_token_logits = next_token_logits + 1e-8
                        # Normalize to valid probability distribution
                        probs = next_token_logits / next_token_logits.sum(dim=-1, keepdim=True)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        probs = F.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Add the predicted token to the input ids
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            # Check if any sequences have reached the EOS token
            eos_mask = (next_token == self.eos_idx).squeeze(-1)
            if eos_mask.any():
                # We don't break the loop because we want to generate all sequences
                # to the full length, but we could implement early stopping here
                pass

        # Decode the generated sequences
        generated_texts = []
        for i in range(num_return_sequences):
            # Get token IDs for this sequence
            token_ids = input_ids[i].cpu().tolist()

            # Find the end of the original prompt in token space
            prompt_ids = self.tokenizer.encode(prompt)
            if token_ids[0] == self.bos_idx and prompt_ids[0] != self.bos_idx:
                prompt_ids = [self.bos_idx] + prompt_ids
            prompt_length = len(prompt_ids)

            # Extract the generated part (remove the prompt, keep only new tokens)
            generated_ids = token_ids[prompt_length:]

            # Find the end token if present
            if self.eos_idx in generated_ids:
                generated_ids = generated_ids[:generated_ids.index(self.eos_idx)]

            # Decode to text
            generated_text = self.tokenizer.decode(generated_ids)

            # Combine with the original prompt
            full_text = prompt + generated_text

            generated_texts.append(full_text)

        if return_attention and attention_maps:
            return generated_texts, attention_maps
        else:
            return generated_texts

    def _generate_with_kv_cache(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        do_sample: bool = True,
    ) -> str:
        """
        Generate text with key-value caching for faster generation.
        
        This is an optimized generation method that caches key-value pairs
        for more efficient inference.
        
        Args:
            prompt: Text prompt to start generation
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Temperature for sampling
            do_sample: Whether to sample from the distribution
            
        Returns:
            Generated text
        """
        # Note: KV caching is not implemented for encoder-decoder models
        # Check if we're working with EncoderDecoderTransformer
        is_encoder_decoder = hasattr(self.model, 'encoder') and hasattr(self.model, 'decoder')
        if is_encoder_decoder:
            # For encoder-decoder models, fall back to regular generation
            result = self.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                num_return_sequences=1
            )
            # Make sure we return a string
            if isinstance(result, list):
                return result[0]
            # Handle case where generate might return (texts, attention_maps) tuple
            elif isinstance(result, tuple) and len(result) > 0 and isinstance(result[0], list):
                return result[0][0]
            return str(result)  # Fallback conversion

        # Encode prompt
        input_ids = self.tokenizer.encode(prompt)

        # Add BOS token if not present
        if len(input_ids) == 0 or input_ids[0] != self.bos_idx:
            input_ids = [self.bos_idx] + input_ids

        # Create tensor and move to device
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)

        # Keep track of generated tokens
        generated = input_ids

        # Initialize past key values (caching)
        past = None

        # Generate auto-regressively
        for _ in range(max_new_tokens):
            # Forward pass
            with torch.no_grad():
                # For the first step, process the entire prompt
                # For subsequent steps, only process the last token
                if past is None:
                    outputs = self.model(input_ids=input_ids)
                else:
                    # Use only the last token as input with past key-values
                    outputs = self.model(input_ids=input_ids[:, -1:], past_key_values=past)

                # Update past for next iteration
                past = outputs.past_key_values if hasattr(outputs, 'past_key_values') else None

                # Get logits for next token prediction
                next_token_logits = outputs.logits[:, -1, :] if hasattr(outputs, "logits") else outputs[:, -1, :]

                # Apply temperature
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature

                # Sample or get argmax
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Add the predicted token to the generated sequence
            generated = torch.cat([generated, next_token], dim=1)

            # Update input_ids for next iteration to only the new token
            input_ids = next_token

            # Stop if we predict the EOS token
            if next_token.item() == self.eos_idx:
                break

        # Decode the generated sequence
        generated_ids = generated[0].cpu().tolist()

        # Find the end token if present
        if self.eos_idx in generated_ids:
            generated_ids = generated_ids[:generated_ids.index(self.eos_idx) + 1]

        # Decode to text
        generated_text = self.tokenizer.decode(generated_ids)

        return generated_text

    def batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        do_sample: bool = True,
    ) -> List[str]:
        """
        Generate text for multiple prompts in batch.
        
        Args:
            prompts: List of text prompts
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Temperature for sampling
            do_sample: Whether to sample from the distribution
            
        Returns:
            List of generated texts
        """
        # For encoder-decoder models, it's more stable to just run generation sequentially
        # Check if we're working with EncoderDecoderTransformer
        is_encoder_decoder = hasattr(self.model, 'encoder') and hasattr(self.model, 'decoder')
        if is_encoder_decoder:
            results = []
            for prompt in prompts:
                # Generate text for each prompt individually
                generated = self.generate(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample
                )
                # Handle both string and list return types
                if isinstance(generated, list):
                    results.append(generated[0])
                else:
                    results.append(str(generated))
            return results

        # Standard batch generation for models supporting input_ids/attention_mask
        # Encode all prompts
        encoded_prompts = []
        max_length = 0

        for prompt in prompts:
            input_ids = self.tokenizer.encode(prompt)

            # Add BOS token if not present
            if len(input_ids) == 0 or input_ids[0] != self.bos_idx:
                input_ids = [self.bos_idx] + input_ids

            encoded_prompts.append(input_ids)
            max_length = max(max_length, len(input_ids))

        # Pad sequences to max length
        padded_prompts = []
        for input_ids in encoded_prompts:
            padded = input_ids + [self.pad_idx] * (max_length - len(input_ids))
            padded_prompts.append(padded)

        # Create tensors
        input_ids = torch.tensor(padded_prompts, dtype=torch.long).to(self.device)
        attention_mask = (input_ids != self.pad_idx).long()

        # Generate in parallel for all prompts
        generated = input_ids.clone()

        # Generate tokens auto-regressively
        for _ in range(max_new_tokens):
            # Forward pass
            with torch.no_grad():
                # Get outputs for standard models
                outputs = self.model(
                    input_ids=generated,
                    attention_mask=attention_mask
                )

                # Get logits
                logits = outputs.logits if hasattr(outputs, "logits") else outputs

                # Focus on the last token prediction
                next_token_logits = logits[:, -1, :]

                # Apply temperature
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature

                # Sample or get argmax
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Add the predicted tokens to the generated sequences
            generated = torch.cat([generated, next_token], dim=1)

            # Update attention mask for next iteration
            attention_mask = torch.cat([
                attention_mask,
                torch.ones_like(next_token)
            ], dim=1)

        # Decode the generated sequences
        generated_texts = []

        for i, prompt in enumerate(prompts):
            # Get token IDs for this sequence
            token_ids = generated[i].cpu().tolist()

            # Find the end of the original prompt
            prompt_length = len(encoded_prompts[i])

            # Extract the generated part (remove the prompt)
            generated_ids = token_ids[prompt_length:]

            # Find the end token if present
            if self.eos_idx in generated_ids:
                generated_ids = generated_ids[:generated_ids.index(self.eos_idx)]

            # Remove padding
            while generated_ids and generated_ids[-1] == self.pad_idx:
                generated_ids.pop()

            # Decode to text
            generated_text = self.tokenizer.decode(generated_ids)

            # Combine with the original prompt
            full_text = prompt + generated_text

            generated_texts.append(full_text)

        return generated_texts

def extract_file_metadata(file_path=__file__):
    """
    Extract structured metadata about this module.
    
    Args:
        file_path: Path to the source file (defaults to current file)
        
    Returns:
        dict: Structured metadata about the module's purpose and components
    """
    return {
        "filename": os.path.basename(file_path),
        "module_purpose": "Provides utilities for text generation using language models",
        "key_classes": [
            {
                "name": "TextGenerator",
                "purpose": "Text generation utilities with various sampling strategies and optimizations",
                "key_methods": [
                    {
                        "name": "generate",
                        "signature": "generate(self, prompt: str, max_new_tokens: int = 50, temperature: float = 1.0, top_k: Optional[int] = None, top_p: Optional[float] = None, do_sample: bool = True, num_return_sequences: int = 1, return_attention: bool = False) -> Union[List[str], Tuple[List[str], List[torch.Tensor]]]",
                        "brief_description": "Generate text from a prompt using various sampling strategies"
                    },
                    {
                        "name": "_generate_with_kv_cache",
                        "signature": "_generate_with_kv_cache(self, prompt: str, max_new_tokens: int = 50, temperature: float = 1.0, do_sample: bool = True) -> str",
                        "brief_description": "Generate text with key-value caching for faster inference"
                    },
                    {
                        "name": "batch_generate",
                        "signature": "batch_generate(self, prompts: List[str], max_new_tokens: int = 50, temperature: float = 1.0, do_sample: bool = True) -> List[str]",
                        "brief_description": "Generate text for multiple prompts in parallel"
                    }
                ],
                "inheritance": "object",
                "dependencies": ["torch", "torch.nn.functional", "numpy"]
            }
        ],
        "external_dependencies": ["torch", "numpy"],
        "complexity_score": 7,  # High complexity due to multiple generation strategies and optimizations
    }
