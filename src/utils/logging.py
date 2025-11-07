"""
Logging utilities for the MultiModal Insight Engine.

This module provides custom logging functionality while ensuring compatibility
with the standard Python logging module.
"""

import logging as std_logging
import os
import sys
from typing import Optional, Union

# Re-export NullHandler from standard logging
NullHandler = std_logging.NullHandler

# Constants
LOG_LEVELS = {
    'DEBUG': std_logging.DEBUG,
    'INFO': std_logging.INFO,
    'WARNING': std_logging.WARNING,
    'ERROR': std_logging.ERROR,
    'CRITICAL': std_logging.CRITICAL
}

DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DEFAULT_LOG_LEVEL = 'INFO'

class LogManager:
    """
    Central logging manager for the application.
    
    This class provides methods for configuring loggers and obtaining logger instances.
    """

    def __init__(self):
        """Initialize the log manager."""
        self.loggers = {}
        self.default_level = LOG_LEVELS.get(os.environ.get('LOG_LEVEL', DEFAULT_LOG_LEVEL), std_logging.INFO)
        self.default_format = os.environ.get('LOG_FORMAT', DEFAULT_LOG_FORMAT)

    def get_logger(self, name: str, level: Optional[Union[str, int]] = None) -> std_logging.Logger:
        """
        Get a logger instance with the specified name.
        
        Args:
            name: Name of the logger
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            
        Returns:
            Logger instance
        """
        if name in self.loggers:
            return self.loggers[name]

        # Create new logger
        logger = std_logging.getLogger(name)

        # Set level
        if level is not None:
            if isinstance(level, str):
                level = LOG_LEVELS.get(level.upper(), self.default_level)
            logger.setLevel(level)
        else:
            logger.setLevel(self.default_level)

        # Add handler if not already present
        if not logger.handlers:
            handler = std_logging.StreamHandler(sys.stdout)
            formatter = std_logging.Formatter(self.default_format)
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        # Store and return
        self.loggers[name] = logger
        return logger

    def configure_file_logging(self, log_dir: str, name: str = 'application') -> None:
        """
        Configure file logging for the application.
        
        Args:
            log_dir: Directory to store log files
            name: Name of the log file (will be used as prefix)
        """
        os.makedirs(log_dir, exist_ok=True)

        # Create file handler
        log_file = os.path.join(log_dir, f"{name}.log")
        file_handler = std_logging.FileHandler(log_file)
        file_handler.setFormatter(std_logging.Formatter(self.default_format))

        # Add to root logger
        root_logger = std_logging.getLogger()
        root_logger.addHandler(file_handler)

        # Log start message
        root_logger.info(f"File logging configured: {log_file}")

# Create singleton instance
log_manager = LogManager()

# Export primary functions
get_logger = log_manager.get_logger
configure_file_logging = log_manager.configure_file_logging

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
        "module_purpose": "Provides custom logging functionality with configurable file and console output",
        "key_classes": [
            {
                "name": "LogManager",
                "purpose": "Central logging manager that configures and provides logger instances",
                "key_methods": [
                    {
                        "name": "get_logger",
                        "signature": "get_logger(self, name: str, level: Optional[Union[str, int]] = None) -> std_logging.Logger",
                        "brief_description": "Creates or retrieves a logger with the specified name and level"
                    },
                    {
                        "name": "configure_file_logging",
                        "signature": "configure_file_logging(self, log_dir: str, name: str = 'application') -> None",
                        "brief_description": "Sets up file logging to the specified directory"
                    }
                ],
                "inheritance": "object",
                "dependencies": ["logging", "os", "sys"]
            }
        ],
        "key_functions": [
            {
                "name": "get_logger",
                "signature": "get_logger(name: str, level: Optional[Union[str, int]] = None) -> std_logging.Logger",
                "brief_description": "Convenience function to get a logger from the singleton manager"
            },
            {
                "name": "configure_file_logging",
                "signature": "configure_file_logging(log_dir: str, name: str = 'application') -> None",
                "brief_description": "Convenience function to configure file logging"
            }
        ],
        "external_dependencies": ["logging"],
        "complexity_score": 4  # Moderate complexity for a utility module
    }
