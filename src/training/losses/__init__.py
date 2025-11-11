from .memory_queue_contrastive_loss import MemoryQueueContrastiveLoss
from .dynamic_temperature_contrastive_loss import DynamicTemperatureContrastiveLoss
from .hard_negative_mining_contrastive_loss import HardNegativeMiningContrastiveLoss
from .contrastive import SimCLRLoss, CLIPLoss
from .multimodal_mixed_contrastive_loss import MultiModalMixedContrastiveLoss
from .decoupled_contrastive_loss import DecoupledContrastiveLoss
from .vicreg_loss import VICRegLoss
from .barlow_twins_loss import BarlowTwinsLoss
from .hybrid_pretrain_vicreg_loss import HybridPretrainVICRegLoss
from .losses import (
    CrossEntropyLoss,
    MeanSquaredError,
)
from .contrastive_learning import (
    nt_xent_loss,
    supervised_contrastive_loss,
    compute_recall_at_k,
)

# Backward compatibility alias
ContrastiveLoss = SimCLRLoss

__all__ = [
    "MemoryQueueContrastiveLoss",
    "DynamicTemperatureContrastiveLoss",
    "HardNegativeMiningContrastiveLoss",
    "ContrastiveLoss",  # Backward compatibility alias
    "SimCLRLoss",
    "CLIPLoss",
    "MultiModalMixedContrastiveLoss",
    "DecoupledContrastiveLoss",
    "VICRegLoss",
    "BarlowTwinsLoss",
    "HybridPretrainVICRegLoss",
    "nt_xent_loss",
    "supervised_contrastive_loss",
    "compute_recall_at_k",
    "CrossEntropyLoss",
    "MeanSquaredError",
]
