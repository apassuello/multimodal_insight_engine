"""
Configuration management utilities for the MultiModal Insight Engine.

This module provides functionality for managing configuration settings
across the application with support for loading from files and environment.
"""

import json
import os
from typing import Any, Dict, Optional


class ConfigManager:
    """Manager for application configuration values."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Optional path to a JSON configuration file
        """
        self.config: Dict[str, Any] = {}

        # Load from file if provided
        if config_path and os.path.exists(config_path):
            self.load_from_file(config_path)

        # Override with environment variables (future extension)

    def load_from_file(self, config_path: str) -> None:
        """
        Load configuration from a JSON file.
        
        Args:
            config_path: Path to the JSON config file
        """
        try:
            with open(config_path, 'r') as f:
                self.config.update(json.load(f))
        except Exception as e:
            print(f"Error loading config from {config_path}: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key to retrieve
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
        """
        return self.config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key to set
            value: Value to set
        """
        self.config[key] = value

    def save_to_file(self, config_path: str) -> None:
        """
        Save current configuration to a JSON file.
        
        Args:
            config_path: Path where to save the config
        """
        try:
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Error saving config to {config_path}: {e}")

# Create a default instance
config_manager = ConfigManager()

# Export primary functions
get_config = config_manager.get
set_config = config_manager.set

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
        "module_purpose": "Provides configuration management utilities with file loading and environment support",
        "key_classes": [
            {
                "name": "ConfigManager",
                "purpose": "Manages application configuration with support for file and environment loading",
                "key_methods": [
                    {
                        "name": "load_from_file",
                        "signature": "load_from_file(self, config_path: str) -> None",
                        "brief_description": "Load configuration from a JSON file"
                    },
                    {
                        "name": "get",
                        "signature": "get(self, key: str, default: Any = None) -> Any",
                        "brief_description": "Get a configuration value with optional default"
                    },
                    {
                        "name": "set",
                        "signature": "set(self, key: str, value: Any) -> None",
                        "brief_description": "Set a configuration value"
                    },
                    {
                        "name": "save_to_file",
                        "signature": "save_to_file(self, config_path: str) -> None",
                        "brief_description": "Save current configuration to a JSON file"
                    }
                ],
                "inheritance": "object",
                "dependencies": ["os", "json"]
            }
        ],
        "key_functions": [
            {
                "name": "get_config",
                "signature": "get_config(key: str, default: Any = None) -> Any",
                "brief_description": "Convenience function to get a config value from the default manager"
            },
            {
                "name": "set_config",
                "signature": "set_config(key: str, value: Any) -> None",
                "brief_description": "Convenience function to set a config value in the default manager"
            }
        ],
        "external_dependencies": ["json"],
        "complexity_score": 3  # Moderate complexity for a configuration manager
    }
