import torch
import numpy as np
import os
from typing import Union, Tuple, List, Dict, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class DataPreprocessor:
    """A class for handling data preprocessing operations."""
    
    def __init__(self, method: str = 'standard'):
        """
        Initialize the preprocessor.
        
        Args:
            method: Scaling method to use ('standard' or 'minmax')
        """
        self.method = method
        self.scaler = StandardScaler() if method == 'standard' else MinMaxScaler()
        self.is_fitted = False
    
    def fit(self, data: Union[torch.Tensor, np.ndarray]) -> None:
        """
        Fit the preprocessor on the data.
        
        Args:
            data: Input data to fit on
        """
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        self.scaler.fit(data)
        self.is_fitted = True
    
    def transform(self, data: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Transform the data using the fitted preprocessor.
        
        Args:
            data: Input data to transform
            
        Returns:
            Transformed data as torch.Tensor
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transforming data")
        
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        
        transformed = self.scaler.transform(data)
        return torch.from_numpy(transformed).float()
    
    def fit_transform(self, data: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Fit the preprocessor and transform the data.
        
        Args:
            data: Input data to fit and transform
            
        Returns:
            Transformed data as torch.Tensor
        """
        self.fit(data)
        return self.transform(data)
    
    def inverse_transform(self, data: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Inverse transform the data back to original scale.
        
        Args:
            data: Transformed data to inverse transform
            
        Returns:
            Original scale data as torch.Tensor
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before inverse transforming data")
        
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        
        original = self.scaler.inverse_transform(data)
        return torch.from_numpy(original).float()

def create_sequences(data: torch.Tensor, seq_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create sequences from time series data.
    
    Args:
        data: Input time series data
        seq_length: Length of each sequence
        
    Returns:
        Tuple of (sequences, targets)
    """
    sequences = []
    targets = []
    
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        targets.append(data[i + seq_length])
    
    return torch.stack(sequences), torch.stack(targets)

def split_data(data: torch.Tensor, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        data: Input data to split
        train_ratio: Ratio of data to use for training
        val_ratio: Ratio of data to use for validation
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    total_size = len(data)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    return train_data, val_data, test_data

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
        "module_purpose": "Provides data preprocessing utilities for time series data and machine learning datasets",
        "key_classes": [
            {
                "name": "DataPreprocessor",
                "purpose": "Handles data preprocessing operations like standardization and normalization",
                "key_methods": [
                    {
                        "name": "fit",
                        "signature": "fit(self, data: Union[torch.Tensor, np.ndarray]) -> None",
                        "brief_description": "Fit the preprocessor on the data"
                    },
                    {
                        "name": "transform",
                        "signature": "transform(self, data: Union[torch.Tensor, np.ndarray]) -> torch.Tensor",
                        "brief_description": "Transform the data using the fitted preprocessor"
                    },
                    {
                        "name": "fit_transform",
                        "signature": "fit_transform(self, data: Union[torch.Tensor, np.ndarray]) -> torch.Tensor",
                        "brief_description": "Fit the preprocessor and transform the data"
                    },
                    {
                        "name": "inverse_transform",
                        "signature": "inverse_transform(self, data: Union[torch.Tensor, np.ndarray]) -> torch.Tensor",
                        "brief_description": "Inverse transform data back to original scale"
                    }
                ],
                "inheritance": "object",
                "dependencies": ["torch", "numpy", "sklearn.preprocessing"]
            }
        ],
        "key_functions": [
            {
                "name": "create_sequences",
                "signature": "create_sequences(data: torch.Tensor, seq_length: int) -> Tuple[torch.Tensor, torch.Tensor]",
                "brief_description": "Create input-target sequences from time series data"
            },
            {
                "name": "split_data",
                "signature": "split_data(data: torch.Tensor, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]",
                "brief_description": "Split data into train, validation, and test sets"
            }
        ],
        "external_dependencies": ["torch", "numpy", "sklearn"],
        "complexity_score": 4  # Moderate complexity with multiple utility functions
    }
