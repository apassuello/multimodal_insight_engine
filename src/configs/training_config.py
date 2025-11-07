# src/configs/training_config.py

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class LossConfig:
    """Configuration for a loss function"""

    name: str
    weight: float = 1.0
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizerConfig:
    """Configuration for optimizer"""

    name: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    """Configuration for learning rate scheduler"""

    name: str = "warmup_cosine"
    warmup_steps: int = 500
    warmup_ratio: float = 0.1
    min_lr_ratio: float = 0.0


@dataclass
class ComponentConfig:
    """Configuration for model component"""

    name: str
    freeze: bool = False
    lr_multiplier: float = 1.0


@dataclass
class StageConfig:
    """Configuration for a training stage"""

    name: str
    losses: List[LossConfig] = field(default_factory=list)
    epochs: int = 10
    batch_size: int = 32
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    components: List[ComponentConfig] = field(default_factory=list)
    early_stopping: bool = True
    patience: int = 5
    evaluation_metrics: List[str] = field(default_factory=lambda: ["val_loss"])
    monitor_metric: str = "val_loss"
    monitor_mode: str = "min"
    clip_grad_norm: Optional[float] = 1.0
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1


@dataclass
class TrainingConfig:
    """Complete training configuration"""

    project_name: str = "MultiModal_Insight_Engine"
    output_dir: str = "outputs"
    seed: int = 42
    stages: List[StageConfig] = field(default_factory=list)
    data_config: Dict[str, Any] = field(default_factory=dict)
    model_config: Dict[str, Any] = field(default_factory=dict)

    def save(self, path: str) -> None:
        """Save configuration to a file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if path.endswith(".json"):
            with open(path, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
        elif path.endswith((".yaml", ".yml")):
            with open(path, "w") as f:
                yaml.dump(self.to_dict(), f)
        else:
            raise ValueError(f"Unsupported file format: {path}")

        logger.info(f"Configuration saved to {path}")

    @classmethod
    def load(cls, path: str) -> "TrainingConfig":
        """Load configuration from a file"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")

        if path.endswith(".json"):
            with open(path, "r") as f:
                config_dict = json.load(f)
        elif path.endswith((".yaml", ".yml")):
            with open(path, "r") as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {path}")

        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        result = {
            "project_name": self.project_name,
            "output_dir": self.output_dir,
            "seed": self.seed,
            "data_config": self.data_config,
            "model_config": self.model_config,
            "stages": [],
        }

        for stage in self.stages:
            stage_dict = {
                "name": stage.name,
                "epochs": stage.epochs,
                "batch_size": stage.batch_size,
                "early_stopping": stage.early_stopping,
                "patience": stage.patience,
                "evaluation_metrics": stage.evaluation_metrics,
                "monitor_metric": stage.monitor_metric,
                "monitor_mode": stage.monitor_mode,
                "clip_grad_norm": stage.clip_grad_norm,
                "mixed_precision": stage.mixed_precision,
                "gradient_accumulation_steps": stage.gradient_accumulation_steps,
                "losses": [
                    {"name": loss.name, "weight": loss.weight, "params": loss.params}
                    for loss in stage.losses
                ],
                "optimizer": {
                    "name": stage.optimizer.name,
                    "lr": stage.optimizer.lr,
                    "weight_decay": stage.optimizer.weight_decay,
                    "betas": stage.optimizer.betas,
                    "eps": stage.optimizer.eps,
                },
                "scheduler": {
                    "name": stage.scheduler.name,
                    "warmup_steps": stage.scheduler.warmup_steps,
                    "warmup_ratio": stage.scheduler.warmup_ratio,
                    "min_lr_ratio": stage.scheduler.min_lr_ratio,
                },
                "components": [
                    {
                        "name": comp.name,
                        "freeze": comp.freeze,
                        "lr_multiplier": comp.lr_multiplier,
                    }
                    for comp in stage.components
                ],
            }
            result["stages"].append(stage_dict)

        return result

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainingConfig":
        """Create config from dictionary"""
        stages = []

        for stage_dict in config_dict.get("stages", []):
            losses = [
                LossConfig(
                    name=loss_dict["name"],
                    weight=loss_dict.get("weight", 1.0),
                    params=loss_dict.get("params", {}),
                )
                for loss_dict in stage_dict.get("losses", [])
            ]

            optimizer_dict = stage_dict.get("optimizer", {})
            optimizer = OptimizerConfig(
                name=optimizer_dict.get("name", "adamw"),
                lr=optimizer_dict.get("lr", 1e-4),
                weight_decay=optimizer_dict.get("weight_decay", 0.01),
                betas=tuple(optimizer_dict.get("betas", (0.9, 0.999))),
                eps=optimizer_dict.get("eps", 1e-8),
            )

            scheduler_dict = stage_dict.get("scheduler", {})
            scheduler = SchedulerConfig(
                name=scheduler_dict.get("name", "warmup_cosine"),
                warmup_steps=scheduler_dict.get("warmup_steps", 500),
                warmup_ratio=scheduler_dict.get("warmup_ratio", 0.1),
                min_lr_ratio=scheduler_dict.get("min_lr_ratio", 0.0),
            )

            components = [
                ComponentConfig(
                    name=comp_dict["name"],
                    freeze=comp_dict.get("freeze", False),
                    lr_multiplier=comp_dict.get("lr_multiplier", 1.0),
                )
                for comp_dict in stage_dict.get("components", [])
            ]

            stage = StageConfig(
                name=stage_dict["name"],
                losses=losses,
                epochs=stage_dict.get("epochs", 10),
                batch_size=stage_dict.get("batch_size", 32),
                optimizer=optimizer,
                scheduler=scheduler,
                components=components,
                early_stopping=stage_dict.get("early_stopping", True),
                patience=stage_dict.get("patience", 5),
                evaluation_metrics=stage_dict.get("evaluation_metrics", ["val_loss"]),
                monitor_metric=stage_dict.get("monitor_metric", "val_loss"),
                monitor_mode=stage_dict.get("monitor_mode", "min"),
                clip_grad_norm=stage_dict.get("clip_grad_norm", 1.0),
                mixed_precision=stage_dict.get("mixed_precision", True),
                gradient_accumulation_steps=stage_dict.get(
                    "gradient_accumulation_steps", 1
                ),
            )

            stages.append(stage)

        return cls(
            project_name=config_dict.get("project_name", "MultiModal_Insight_Engine"),
            output_dir=config_dict.get("output_dir", "outputs"),
            seed=config_dict.get("seed", 42),
            stages=stages,
            data_config=config_dict.get("data_config", {}),
            model_config=config_dict.get("model_config", {}),
        )

    @classmethod
    def create_default_multistage_config(cls) -> "TrainingConfig":
        """Create a default multistage training configuration"""
        config = TrainingConfig()

        # Stage 1: Modality-specific learning
        stage1 = StageConfig(
            name="modality_specific_learning",
            epochs=5,
            batch_size=64,
            optimizer=OptimizerConfig(lr=5e-5),
            losses=[
                LossConfig(
                    name="contrastive_loss", weight=1.0, params={"temperature": 0.07}
                ),
                LossConfig(name="decorrelation_loss", weight=0.25, params={}),
            ],
            components=[
                ComponentConfig(name="vision_model", freeze=True, lr_multiplier=0.1),
                ComponentConfig(name="text_model", freeze=True, lr_multiplier=0.1),
                ComponentConfig(
                    name="vision_projection", freeze=False, lr_multiplier=1.0
                ),
                ComponentConfig(
                    name="text_projection", freeze=False, lr_multiplier=1.0
                ),
                ComponentConfig(name="cross_attention", freeze=True, lr_multiplier=0.0),
            ],
            monitor_metric="val_alignment_score",
            monitor_mode="max",
        )

        # Stage 2: Cross-modal fusion
        stage2 = StageConfig(
            name="cross_modal_fusion",
            epochs=8,
            batch_size=64,
            optimizer=OptimizerConfig(lr=1e-4),
            losses=[
                LossConfig(
                    name="memory_queue_contrastive_loss",
                    weight=1.0,
                    params={"queue_size": 8192, "momentum": 0.99, "temperature": 0.07},
                ),
                LossConfig(name="decorrelation_loss", weight=0.25, params={}),
            ],
            components=[
                ComponentConfig(name="vision_model", freeze=True, lr_multiplier=0.0),
                ComponentConfig(name="text_model", freeze=True, lr_multiplier=0.0),
                ComponentConfig(
                    name="vision_projection", freeze=False, lr_multiplier=0.5
                ),
                ComponentConfig(
                    name="text_projection", freeze=False, lr_multiplier=0.5
                ),
                ComponentConfig(
                    name="cross_attention", freeze=False, lr_multiplier=1.0
                ),
            ],
            monitor_metric="val_alignment_score",
            monitor_mode="max",
        )

        # Stage 3: Fine-tuning
        stage3 = StageConfig(
            name="end_to_end_fine_tuning",
            epochs=10,
            batch_size=32,
            optimizer=OptimizerConfig(lr=5e-6),
            losses=[
                LossConfig(
                    name="hard_negative_mining_contrastive_loss",
                    weight=1.0,
                    params={"mining_ratio": 0.5, "temperature": 0.05},
                ),
                LossConfig(
                    name="feature_consistency_loss",
                    weight=0.5,
                    params={"distance": "cosine"},
                ),
                LossConfig(name="decorrelation_loss", weight=0.1, params={}),
            ],
            components=[
                ComponentConfig(name="vision_model", freeze=False, lr_multiplier=0.01),
                ComponentConfig(name="text_model", freeze=False, lr_multiplier=0.01),
                ComponentConfig(
                    name="vision_projection", freeze=False, lr_multiplier=0.1
                ),
                ComponentConfig(
                    name="text_projection", freeze=False, lr_multiplier=0.1
                ),
                ComponentConfig(
                    name="cross_attention", freeze=False, lr_multiplier=0.5
                ),
            ],
            monitor_metric="val_recall_at_5",
            monitor_mode="max",
            gradient_accumulation_steps=2,
        )

        config.stages = [stage1, stage2, stage3]

        # Example data and model configuration
        config.data_config = {
            "dataset": "flickr30k",
            "train_split_ratio": 0.8,
            "val_split_ratio": 0.1,
            "test_split_ratio": 0.1,
            "image_size": 224,
            "max_text_length": 77,
        }

        config.model_config = {
            "vision_model": "google/vit-base-patch16-224",
            "text_model": "bert-base-uncased",
            "projection_dim": 512,
            "num_cross_attention_heads": 8,
            "cross_attention_dropout": 0.1,
        }

        return config
