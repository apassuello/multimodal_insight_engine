# src/optimization/pruning.py
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import Dict, List, Optional, Union, Any
import logging

class PruningConfig:
    """
    Configuration class for model pruning settings.
    
    This class centralizes parameters for different pruning approaches,
    making it easier to experiment with various configurations.
    """
    
    def __init__(
        self,
        method: str = "magnitude",  # "magnitude", "structured", "l1_unstructured", etc.
        amount: Union[float, int] = 0.2,  # Amount to prune (percentage or absolute)
        dim: Optional[int] = None,  # Dimension for structured pruning
        n_iterations: int = 1,  # Number of pruning iterations
        pruning_dims: Optional[List[str]] = None,  # Parameters to prune
        sparsity_distribution: str = "uniform",  # How to distribute sparsity
        reinitialize: bool = False,  # Whether to reinitialize pruned weights
    ):
        """
        Initialize pruning configuration.
        
        Args:
            method: Pruning method to use
            amount: Amount to prune (between 0 and 1 for percentage)
            dim: Dimension for structured pruning
            n_iterations: Number of pruning iterations (for iterative pruning)
            pruning_dims: Parameters to prune (if None, prune all weights)
            sparsity_distribution: How to distribute sparsity across layers
            reinitialize: Whether to reinitialize pruned weights
        """
        self.method = method
        self.amount = amount
        self.dim = dim
        self.n_iterations = n_iterations
        self.pruning_dims = pruning_dims or ["weight"]
        self.sparsity_distribution = sparsity_distribution
        self.reinitialize = reinitialize
    
    def __str__(self) -> str:
        """String representation of the configuration."""
        return (
            f"PruningConfig(method={self.method}, "
            f"amount={self.amount}, "
            f"iterations={self.n_iterations}, "
            f"distribution={self.sparsity_distribution})"
        )


class ModelPruner:
    """
    Implements various pruning techniques for neural networks.
    
    Pruning removes weights from a neural network to reduce its size and
    potentially improve its inference speed, with minimal impact on accuracy.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[PruningConfig] = None,
    ):
        """
        Initialize the model pruner.
        
        Args:
            model: The model to prune
            config: Pruning configuration
        """
        self.model = model
        self.config = config or PruningConfig()
        self.original_state_dict = {
            k: v.clone() for k, v in model.state_dict().items()
        }
        
        # Track pruning statistics
        self.pruning_history = []
    
    def prune_model(self) -> nn.Module:
        """
        Apply pruning to the model.
        
        Returns:
            Pruned model
        """
        if self.config.method == "magnitude":
            return self._apply_magnitude_pruning()
        elif self.config.method == "structured":
            return self._apply_structured_pruning()
        elif self.config.method == "iterative_magnitude":
            return self._apply_iterative_pruning()
        else:
            raise ValueError(f"Unsupported pruning method: {self.config.method}")
    
    def _apply_magnitude_pruning(self) -> nn.Module:
        """
        Apply magnitude-based unstructured pruning.
        
        This method prunes the smallest weights by absolute value.
        
        Returns:
            Pruned model
        """
        prunable_modules = []
        
        # Identify prunable modules
        for name, module in self.model.named_modules():
            for param_name in self.config.pruning_dims:
                if hasattr(module, param_name):
                    prunable_modules.append((name, module))
                    break
        
        # Apply pruning
        for name, module in prunable_modules:
            for param_name in self.config.pruning_dims:
                if hasattr(module, param_name):
                    prune.l1_unstructured(
                        module=module,
                        name=param_name,
                        amount=self.config.amount,
                    )
        
        # Store pruning statistics
        self.pruning_history.append(self._calculate_sparsity())
        
        return self.model
    
    def _apply_structured_pruning(self) -> nn.Module:
        """
        Apply structured pruning along specified dimensions.
        
        Structured pruning removes entire rows, columns, or other structures,
        which can better leverage hardware acceleration.
        
        Returns:
            Pruned model
        """
        if self.config.dim is None:
            raise ValueError("Structured pruning requires specifying a dimension")
        
        prunable_modules = []
        
        # Identify prunable modules
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                prunable_modules.append((name, module))
        
        # Apply structured pruning
        for name, module in prunable_modules:
            prune.ln_structured(
                module=module,
                name="weight",
                amount=self.config.amount,
                n=2,  # L2 norm
                dim=self.config.dim  # 0 for rows (output features), 1 for columns (input features)
            )
        
        # Store pruning statistics
        self.pruning_history.append(self._calculate_sparsity())
        
        return self.model
    
    def _apply_iterative_pruning(self) -> nn.Module:
        """
        Apply iterative magnitude pruning.
        
        Iterative pruning gradually increases sparsity over multiple iterations,
        often resulting in better accuracy than one-shot pruning.
        
        Returns:
            Pruned model
        """
        iterations = self.config.n_iterations
        target_sparsity = self.config.amount
        
        # Calculate sparsity for each iteration
        sparsities = [
            1.0 - (1.0 - target_sparsity) * ((iterations - i) / iterations) ** 3
            for i in range(iterations)
        ]
        
        for i, sparsity in enumerate(sparsities):
            logging.info(f"Pruning iteration {i+1}/{iterations} with sparsity {sparsity:.4f}")
            
            # Create a temporary config for this iteration
            iter_config = PruningConfig(
                method="magnitude",
                amount=sparsity,
                pruning_dims=self.config.pruning_dims,
            )
            
            # Apply pruning for this iteration
            temp_pruner = ModelPruner(self.model, config=iter_config)
            temp_pruner._apply_magnitude_pruning()
            
            # Store pruning statistics
            self.pruning_history.append(self._calculate_sparsity())
            
            # Fine-tune the model between iterations (would be implemented in practice)
            # self._fine_tune_pruned_model()
        
        return self.model
    
    def _calculate_sparsity(self) -> Dict[str, float]:
        """
        Calculate model sparsity after pruning.
        
        Returns:
            Dictionary with sparsity statistics
        """
        total_params = 0
        zero_params = 0
        
        for name, param in self.model.named_parameters():
            if any(dim in name for dim in self.config.pruning_dims):
                total_params += param.numel()
                zero_params += (param == 0).sum().item()
        
        return {
            "sparsity": zero_params / total_params if total_params > 0 else 0,
            "total_params": total_params,
            "zero_params": zero_params,
        }
    
    def restore_model(self):
        """Restore the model to its original unpruned state."""
        self.model.load_state_dict(self.original_state_dict)
        self.pruning_history = []
    
    def get_pruning_info(self) -> Dict[str, Any]:
        """
        Get information about pruning results.
        
        Returns:
            Dictionary with pruning information
        """
        if not self.pruning_history:
            return {"error": "Model has not been pruned yet. Call prune_model() first."}
        
        return {
            "method": self.config.method,
            "final_sparsity": self.pruning_history[-1]["sparsity"],
            "iterations": len(self.pruning_history),
            "history": self.pruning_history,
        }