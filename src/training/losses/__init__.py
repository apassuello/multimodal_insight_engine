from .contrastive import SimCLRLoss, CLIPLoss, MoCoLoss, HardNegativeLoss
from .dynamic_temperature_contrastive_loss import DynamicTemperatureContrastiveLoss
from .multimodal import MixedMultimodalLoss
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

# Backward compatibility aliases
ContrastiveLoss = SimCLRLoss
MultiModalMixedContrastiveLoss = MixedMultimodalLoss
MemoryQueueContrastiveLoss = MoCoLoss
HardNegativeMiningContrastiveLoss = HardNegativeLoss

__all__ = [
    "MemoryQueueContrastiveLoss",  # Backward compatibility alias
    "MoCoLoss",
    "DynamicTemperatureContrastiveLoss",
    "HardNegativeMiningContrastiveLoss",  # Backward compatibility alias
    "HardNegativeLoss",
    "ContrastiveLoss",  # Backward compatibility alias
    "SimCLRLoss",
    "CLIPLoss",
    "MultiModalMixedContrastiveLoss",  # Backward compatibility alias
    "MixedMultimodalLoss",
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
