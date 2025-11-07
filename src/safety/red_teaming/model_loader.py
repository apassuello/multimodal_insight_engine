# src/safety/red_teaming/model_loader.py

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelLoader:
    """
    Utility for loading different types of language models for red teaming.
    
    This class provides utilities to load both locally trained PyTorch models
    and pre-trained models from Hugging Face, and wrap them with a consistent
    interface for use in red teaming exercises.
    """

    def __init__(
        self,
        local_models_dir: str = "./models/pretrained/",
        device: Optional[str] = None,
        max_length: int = 1024,
        temperature: float = 0.7,
        verbose: bool = False
    ):
        """
        Initialize the model loader.
        
        Args:
            local_models_dir: Directory where local models are stored
            device: Device to load models on ('cpu', 'cuda', 'mps', or None for auto-detection)
            max_length: Maximum sequence length for generation
            temperature: Sampling temperature for text generation
            verbose: Whether to print detailed information during loading and inference
        """
        self.local_models_dir = Path(local_models_dir)
        self.verbose = verbose

        # Create directory if it doesn't exist
        os.makedirs(self.local_models_dir, exist_ok=True)

        # Determine device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        if self.verbose:
            print(f"Using device: {self.device}")

        # Generation parameters
        self.max_length = max_length
        self.temperature = temperature

        # Cache for loaded models
        self.loaded_models = {}

    def load_model(
        self,
        model_name: str,
        is_local: bool = None
    ) -> Callable[[str], str]:
        """
        Load a model by name and return a callable function.
        
        Args:
            model_name: Name of the model to load (local path or HF identifier)
            is_local: Force interpretation as local model or HF model
                      (if None, will try to detect automatically)
            
        Returns:
            Function that takes text input and returns model output
        """
        # Check if model is already loaded
        if model_name in self.loaded_models:
            print(f"Using cached model: {model_name}")
            return self.loaded_models[model_name]

        # Determine if model is local or from Hugging Face
        if is_local is None:
            # Check if model exists in local directory
            local_path = self.local_models_dir / model_name
            is_local = local_path.exists()
            print(f"Listing available local models in: {self.local_models_dir}")
            print(f"Listing available local model_path in: {local_path}")

        # Load the appropriate model type
        if is_local:
            model_func = self._load_local_model(model_name)
        else:
            model_func = self._load_huggingface_model(model_name)

        # Cache the loaded model
        self.loaded_models[model_name] = model_func

        return model_func

    def _load_local_model(self, model_name: str) -> Callable[[str], str]:
        """
        Load a locally trained model or local Hugging Face model.
        
        Args:
            model_name: Name of the model directory in local_models_dir
            
        Returns:
            Function that takes text input and returns model output
        """
        model_path = self.local_models_dir / model_name
        print(f"Loading local model from: {model_path}")

        # Check if this is a Hugging Face format model
        if (model_path / "config.json").exists() and (model_path / "tokenizer.json").exists():
            print("Detected Hugging Face format model")
            return self._load_huggingface_model(str(model_path))

        # Otherwise, try to load as a custom PyTorch model
        print("Attempting to load as custom PyTorch model")
        # Check for config file to determine model type
        config_path = model_path / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)
                model_type = config.get("model_type", "transformer")
        else:
            # Default to transformer type if no config
            model_type = "transformer"

        # Load model based on type
        if model_type == "transformer":
            return self._load_local_transformer(model_path)
        elif model_type == "encoder_decoder":
            return self._load_local_encoder_decoder(model_path)
        else:
            raise ValueError(f"Unsupported local model type: {model_type}")

    def _load_local_transformer(self, model_path: Path) -> Callable[[str], str]:
        """Load a local transformer model."""
        # Import here to avoid circular imports
        from src.models.transformer import EncoderDecoderTransformer, Transformer

        # Check which model class to use
        encoder_only_path = model_path / "encoder_only"

        if encoder_only_path.exists() or not (model_path / "decoder.pt").exists():
            # Load encoder-only transformer
            print("Loading encoder-only transformer model")
            model = Transformer()
            model.load(str(model_path / "model.pt"), map_location=self.device)
        else:
            # Load encoder-decoder transformer
            print("Loading encoder-decoder transformer model")
            model = EncoderDecoderTransformer(src_vocab_size=10000, tgt_vocab_size=10000)  # Placeholder sizes
            model.load(str(model_path / "model.pt"), map_location=self.device)

        # Set model to evaluation mode
        model.eval()
        model.to(torch.device(self.device))

        # Load tokenizer
        tokenizer_path = model_path / "tokenizer"
        if tokenizer_path.exists():
            # Import tokenizer
            from src.data.tokenization import BPETokenizer
            tokenizer = BPETokenizer.from_pretrained(str(tokenizer_path))
        else:
            # Create a simple tokenizer function
            tokenizer = lambda text: text.split()
            tokenizer.encode = lambda text: [ord(c) for c in text]
            tokenizer.decode = lambda ids: ''.join(chr(i) for i in ids)

        # Create a wrapped function for inference
        def model_func(prompt: str) -> str:
            try:
                with torch.no_grad():
                    # Tokenize input
                    if hasattr(tokenizer, "encode"):
                        input_ids = tokenizer.encode(prompt)
                        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
                    else:
                        # Fallback for simple tokenizer
                        input_tensor = torch.tensor([[ord(c) for c in prompt]], dtype=torch.long).to(self.device)

                    # Generate output
                    if isinstance(model, EncoderDecoderTransformer):
                        # For encoder-decoder, use its generation method
                        output_ids = model.generate(
                            input_tensor,
                            max_len=self.max_length,
                            bos_token_id=1,  # Assuming BOS token ID is 1
                            eos_token_id=2,  # Assuming EOS token ID is 2
                            temperature=self.temperature
                        )
                        output_ids = output_ids[0].tolist()  # Take first sequence
                    else:
                        # For encoder-only, just use forward pass
                        logits = model(input_tensor)
                        # Simple greedy decoding
                        output_ids = torch.argmax(logits, dim=-1)[0].tolist()

                    # Decode output
                    if hasattr(tokenizer, "decode"):
                        output_text = tokenizer.decode(output_ids)
                    else:
                        # Fallback for simple tokenizer
                        output_text = ''.join(chr(i) for i in output_ids if i < 128)  # ASCII only

                    return output_text
            except Exception as e:
                return f"Error generating response: {str(e)}"

        return model_func

    def _load_local_encoder_decoder(self, model_path: Path) -> Callable[[str], str]:
        """Load a local encoder-decoder model."""
        # This implementation would be similar to _load_local_transformer
        # but tailored to your specific encoder-decoder architecture
        # For now, we'll use a placeholder implementation

        def model_func(prompt: str) -> str:
            try:
                return f"[Output from local encoder-decoder model for: {prompt[:50]}...]"
            except Exception as e:
                return f"Error generating response: {str(e)}"

        return model_func

    def _load_huggingface_model(self, model_name: str) -> Callable[[str], str]:
        """
        Load a model from Hugging Face (remote or local).
        
        Args:
            model_name: Hugging Face model identifier or local path
            
        Returns:
            Function that takes text input and returns model output
        """
        if self.verbose:
            print(f"\nLoading Hugging Face model: {model_name}")
            print(f"Model path exists: {os.path.exists(model_name)}")
            if os.path.exists(model_name):
                print("Contents of model directory:")
                for file in os.listdir(model_name):
                    print(f"  - {file}")

        try:
            if self.verbose:
                print("\nLoading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            if self.verbose:
                print("Tokenizer loaded successfully")

            if self.verbose:
                print("\nLoading model...")
                print(f"Device: {self.device}")
                print(f"Using device_map: auto")

            # Memory-efficient loading options
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                local_files_only=True,
                torch_dtype=torch.float16 if self.device == "cuda" else None,
                device_map="auto",  # Automatically handle model placement
                low_cpu_mem_usage=True,
                offload_folder="offload",  # Enable disk offloading
                offload_state_dict=True,  # Offload weights to disk when not in use
                max_memory={0: "28GB"} if self.device == "mps" else None  # Limit GPU memory usage
            )
            if self.verbose:
                print("Model loaded successfully")

            # Set to evaluation mode
            model.eval()
            if self.verbose:
                print("Model set to evaluation mode")

            # Create a wrapped function for inference
            def model_func(prompt: str) -> str:
                try:
                    if self.verbose:
                        print(f"\nProcessing prompt: {prompt[:50]}...")
                    with torch.no_grad():
                        # Tokenize input
                        if self.verbose:
                            print("Tokenizing input...")
                        inputs = tokenizer(prompt, return_tensors="pt")
                        if self.verbose:
                            print(f"Input shape: {inputs['input_ids'].shape}")

                        # Move inputs to device
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        if self.verbose:
                            print(f"Inputs moved to device: {self.device}")

                        # Generate output
                        if self.verbose:
                            print("Generating output...")
                        outputs = model.generate(
                            **inputs,
                            max_length=min(self.max_length, tokenizer.model_max_length),
                            temperature=self.temperature,
                            do_sample=True if self.temperature > 0 else False,
                            pad_token_id=tokenizer.eos_token_id,
                            num_return_sequences=1,
                            early_stopping=True,
                            min_length=10,  # Ensure minimum response length
                            repetition_penalty=1.2,  # Reduce repetition
                            no_repeat_ngram_size=3,  # Prevent repeating phrases
                            length_penalty=1.0,  # Encourage longer responses
                            top_k=50,  # Limit vocabulary choices
                            top_p=0.95  # Nucleus sampling
                        )
                        if self.verbose:
                            print(f"Output shape: {outputs.shape}")

                        # Decode output
                        if self.verbose:
                            print("Decoding output...")
                        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

                        # For some models, we need to remove the prompt from the output
                        if output_text.startswith(prompt):
                            output_text = output_text[len(prompt):].strip()

                        # Ensure non-empty response
                        if not output_text or output_text.strip() == "":
                            output_text = "I apologize, but I cannot generate a response to that prompt."

                        if self.verbose:
                            print("\nModel Output:")
                            print("-" * 50)
                            print(output_text[:500] + "..." if len(output_text) > 500 else output_text)
                            print("-" * 50)
                            print("Generation complete")

                        return output_text
                except Exception as e:
                    if self.verbose:
                        print(f"Error during inference: {str(e)}")
                    return f"Error generating response: {str(e)}"

            return model_func

        except Exception as e:
            error_msg = str(e)
            if self.verbose:
                print(f"\nError loading model from Hugging Face: {error_msg}")
                print(f"Error type: {type(e).__name__}")

            # Return a function that explains the error
            def error_func(prompt: str) -> str:
                return f"Error loading model '{model_name}': {error_msg}"

            return error_func

    def list_available_local_models(self) -> list:
        """
        List all available local models.
        
        Returns:
            List of model names
        """
        print(f"Listing available local models in: {self.local_models_dir}")
        if not self.local_models_dir.exists():
            return []

        return [d.name for d in self.local_models_dir.iterdir() if d.is_dir()]

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get information about a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model information
        """
        model_path = self.local_models_dir / model_name
        print(f"Listing available local models in: {self.local_models_dir}")
        print(f"Listing available local model_path in: {model_path}")

        if not model_path.exists():
            # Try to get info from Hugging Face
            try:
                from huggingface_hub import model_info
                info = model_info(model_name)
                return {
                    "name": model_name,
                    "source": "huggingface",
                    "downloads": info.downloads,
                    "likes": info.likes,
                    "tags": info.tags,
                    "pipeline_tag": info.pipeline_tag
                }
            except Exception as e:
                return {
                    "name": model_name,
                    "error": str(e)
                }

        # Get info from local model
        info = {
            "name": model_name,
            "source": "local",
            "path": str(model_path),
        }

        # Read config if available
        config_path = model_path / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                info["config"] = json.load(f)

        return info


def load_model(
    model_name: str,
    is_local: bool = None,
    device: str = None,
    max_length: int = 1024,
    temperature: float = 0.7,
    verbose: bool = False
) -> Callable[[str], str]:
    """
    Load a model by name and return a callable function.
    
    This is a convenience wrapper around ModelLoader for simpler usage.
    
    Args:
        model_name: Name of the model to load (local path or HF identifier)
        is_local: Force interpretation as local model or HF model
                  (if None, will try to detect automatically)
        device: Device to load model on ('cpu', 'cuda', 'mps', or None for auto-detection)
        max_length: Maximum sequence length for generation
        temperature: Sampling temperature for text generation
        verbose: Whether to print detailed information during loading and inference
        
    Returns:
        Function that takes text input and returns model output
    """
    loader = ModelLoader(
        device=device,
        max_length=max_length,
        temperature=temperature,
        verbose=verbose
    )

    return loader.load_model(model_name, is_local=is_local)
