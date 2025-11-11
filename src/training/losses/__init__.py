from .contrastive import SimCLRLoss, CLIPLoss, MoCoLoss, HardNegativeLoss, DynamicTemperatureLoss, DecoupledLoss
from .multimodal import MixedMultimodalLoss
from .supervised import SupervisedContrastiveLoss as SupervisedLoss
from .self_supervised import VICRegLoss, BarlowTwinsLoss
from .wrappers import CombinedLoss, MultitaskLoss
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
DynamicTemperatureContrastiveLoss = DynamicTemperatureLoss
DecoupledContrastiveLoss = DecoupledLoss
SupervisedContrastiveLoss = SupervisedLoss

__all__ = [
    "MemoryQueueContrastiveLoss",  # Backward compatibility alias
    "MoCoLoss",
    "DynamicTemperatureContrastiveLoss",  # Backward compatibility alias
    "DynamicTemperatureLoss",
    "HardNegativeMiningContrastiveLoss",  # Backward compatibility alias
    "HardNegativeLoss",
    "ContrastiveLoss",  # Backward compatibility alias
    "SimCLRLoss",
    "CLIPLoss",
    "MultiModalMixedContrastiveLoss",  # Backward compatibility alias
    "MixedMultimodalLoss",
    "DecoupledContrastiveLoss",  # Backward compatibility alias
    "DecoupledLoss",
    "SupervisedContrastiveLoss",  # Backward compatibility alias
    "SupervisedLoss",
    "CombinedLoss",
    "MultitaskLoss",
    "VICRegLoss",
    "BarlowTwinsLoss",
    "HybridPretrainVICRegLoss",
    "nt_xent_loss",
    "supervised_contrastive_loss",
    "compute_recall_at_k",
    "CrossEntropyLoss",
    "MeanSquaredError",
]
