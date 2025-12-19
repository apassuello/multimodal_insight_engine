#!/usr/bin/env python3
"""
Download models for Constitutional AI Demo dual model architecture.
Run this script to pre-download Qwen2-1.5B-Instruct and Phi-2.
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def download_model(model_id: str, name: str):
    """Download a model from Hugging Face."""
    print(f"\n{'='*60}")
    print(f"Downloading {name}")
    print(f"Model ID: {model_id}")
    print(f"{'='*60}\n")

    try:
        # Download tokenizer
        print(f"[1/2] Downloading tokenizer for {name}...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        print(f"✓ Tokenizer downloaded\n")

        # Download model
        print(f"[2/2] Downloading model for {name}...")
        print(f"(This may take a few minutes depending on your connection)\n")

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        print(f"✓ Model downloaded\n")

        # Get size info
        num_params = sum(p.numel() for p in model.parameters())
        num_params_b = num_params / 1_000_000_000
        print(f"✓ {name} ready!")
        print(f"  Parameters: {num_params_b:.2f}B")
        print(f"  Location: ~/.cache/huggingface/hub/")

        # Clean up to save memory
        del model
        del tokenizer

        return True

    except Exception as e:
        print(f"✗ Error downloading {name}: {e}")
        return False


def main():
    print("\n" + "="*60)
    print("Constitutional AI Demo - Model Downloader")
    print("="*60)
    print("\nThis will download two models:")
    print("  1. Qwen2-1.5B-Instruct (~3GB) - For evaluation")
    print("  2. Phi-2 (~5.4GB) - For generation/training")
    print("\nTotal download: ~8.4GB")
    print("Models will be cached in: ~/.cache/huggingface/hub/")
    print("\n" + "="*60 + "\n")

    response = input("Continue? [y/N]: ").strip().lower()
    if response not in ['y', 'yes']:
        print("Download cancelled.")
        return

    # Download Qwen2-1.5B-Instruct
    success1 = download_model(
        model_id="Qwen/Qwen2-1.5B-Instruct",
        name="Qwen2-1.5B-Instruct (Evaluation Model)"
    )

    # Download Phi-2
    success2 = download_model(
        model_id="microsoft/phi-2",
        name="Phi-2 (Generation Model)"
    )

    print("\n" + "="*60)
    if success1 and success2:
        print("✓ All models downloaded successfully!")
        print("\nYou can now run the demo:")
        print("  python3 run_demo.py")
        print("\nThen in the UI:")
        print("  1. Open 'Advanced: Dual Model Architecture' accordion")
        print("  2. Load Qwen2-1.5B-Instruct as Evaluation Model")
        print("  3. Load Phi-2 as Generation Model")
        print("  4. Run training and see the improvements!")
    else:
        print("✗ Some models failed to download")
        print("You can try again or load them from the UI")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
