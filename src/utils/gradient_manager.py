"""Gradient Management :

- GradientHandler: Prevents training instability and modality dominance
- GradientAccumulator: Enables larger effective batch sizes
- GradientScaler: Enables stable mixed precision training
- ModalityGradientBalancer: Balances gradients between modalities. Prevents one modality from dominating training
"""
