# utils/list_models.py

import argparse
import os
import sys


# Add src directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safety.red_teaming.model_loader import ModelLoader
from src.utils.logging import get_logger


logger = get_logger(__name__)


def main(args):
    """List available models and their information."""
    # Create model loader
    loader = ModelLoader()

    if args.list_local:
        logger.info("=== Available Local Models ===")
        local_models = loader.list_available_local_models()

        if not local_models:
            logger.info("No local models found in ./data/pretrained/")
        else:
            for model_name in local_models:
                info = loader.get_model_info(model_name)
                logger.info(f"- {model_name}")

                # Print basic info
                logger.info(f"  Path: {info.get('path', 'N/A')}")

                # Print config if available
                if "config" in info:
                    config = info["config"]
                    if "model_type" in config:
                        logger.info(f"  Type: {config['model_type']}")
                    if "d_model" in config:
                        logger.info(f"  Dimensions: {config['d_model']}")
                    if "num_layers" in config:
                        logger.info(f"  Layers: {config['num_layers']}")

                logger.info("")

    if args.list_hf:
        try:
            from huggingface_hub import list_models

            logger.info("=== Popular Hugging Face Models ===")

            models = list_models(
                filter="text-generation", sort="downloads", direction=-1, limit=args.limit
            )

            for model in models:
                logger.info(f"- {model.id}")
                logger.info(f"  Downloads: {model.downloads:,}")
                logger.info(f"  Likes: {model.likes:,}")
                if model.pipeline_tag:
                    logger.info(f"  Pipeline: {model.pipeline_tag}")
                logger.info("")

        except ImportError:
            logger.info("Error: huggingface_hub package not installed.")
            logger.info("Install it with: pip install huggingface_hub")

    if args.info:
        logger.info(f"=== Model Information: {args.info} ===")
        info = loader.get_model_info(args.info)

        for key, value in info.items():
            if key == "config":
                logger.info("Config:")
                for config_key, config_value in value.items():
                    logger.info(f"  {config_key}: {config_value}")
            else:
                logger.info(f"{key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="List available models for red teaming")

    # List options
    parser.add_argument(
        "--list-local", action="store_true", help="List local models in ./data/pretrained/"
    )
    parser.add_argument("--list-hf", action="store_true", help="List popular Hugging Face models")
    parser.add_argument(
        "--limit", type=int, default=10, help="Limit the number of Hugging Face models listed"
    )

    # Model info
    parser.add_argument("--info", type=str, help="Get detailed information about a specific model")

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
                "brief_description": "List available models and their information based on provided arguments",
            }
        ],
        "external_dependencies": ["argparse", "huggingface_hub"],
        "complexity_score": 4,  # Moderate complexity
    }
