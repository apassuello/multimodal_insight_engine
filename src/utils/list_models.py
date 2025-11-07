# utils/list_models.py

import argparse
import os
import sys

# Add src directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safety.red_teaming.model_loader import ModelLoader


def main(args):
    """List available models and their information."""
    # Create model loader
    loader = ModelLoader()

    if args.list_local:
        print("=== Available Local Models ===")
        local_models = loader.list_available_local_models()

        if not local_models:
            print("No local models found in ./data/pretrained/")
        else:
            for model_name in local_models:
                info = loader.get_model_info(model_name)
                print(f"- {model_name}")

                # Print basic info
                print(f"  Path: {info.get('path', 'N/A')}")

                # Print config if available
                if "config" in info:
                    config = info["config"]
                    if "model_type" in config:
                        print(f"  Type: {config['model_type']}")
                    if "d_model" in config:
                        print(f"  Dimensions: {config['d_model']}")
                    if "num_layers" in config:
                        print(f"  Layers: {config['num_layers']}")

                print()

    if args.list_hf:
        try:
            from huggingface_hub import list_models

            print("=== Popular Hugging Face Models ===")

            models = list_models(
                filter="text-generation",
                sort="downloads",
                direction=-1,
                limit=args.limit
            )

            for model in models:
                print(f"- {model.id}")
                print(f"  Downloads: {model.downloads:,}")
                print(f"  Likes: {model.likes:,}")
                if model.pipeline_tag:
                    print(f"  Pipeline: {model.pipeline_tag}")
                print()

        except ImportError:
            print("Error: huggingface_hub package not installed.")
            print("Install it with: pip install huggingface_hub")

    if args.info:
        print(f"=== Model Information: {args.info} ===")
        info = loader.get_model_info(args.info)

        for key, value in info.items():
            if key == "config":
                print("Config:")
                for config_key, config_value in value.items():
                    print(f"  {config_key}: {config_value}")
            else:
                print(f"{key}: {value}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="List available models for red teaming")

    # List options
    parser.add_argument(
        "--list-local",
        action="store_true",
        help="List local models in ./data/pretrained/"
    )
    parser.add_argument(
        "--list-hf",
        action="store_true",
        help="List popular Hugging Face models"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Limit the number of Hugging Face models listed"
    )

    # Model info
    parser.add_argument(
        "--info",
        type=str,
        help="Get detailed information about a specific model"
    )

    # Default behavior
    args = parser.parse_args()
    if not (args.list_local or args.list_hf or args.info):
        args.list_local = True
        args.list_hf = True

    main(args)

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
        "module_purpose": "Provides utilities for listing and retrieving information about available models",
        "key_functions": [
            {
                "name": "main",
                "signature": "main(args)",
                "brief_description": "List available models and their information based on provided arguments"
            }
        ],
        "external_dependencies": ["argparse", "huggingface_hub"],
        "complexity_score": 4  # Moderate complexity
    }
