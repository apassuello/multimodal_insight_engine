#!/usr/bin/env python
"""
Minimal Contrastive Learning Test

This script tests whether contrastive learning fundamentals work
WITHOUT all the complexity of the full training pipeline.

Expected behavior:
- Diagonal similarity should increase from ~0.0 to >0.6
- Gap between diagonal and off-diagonal should grow
- Loss should decrease

If this doesn't work, your loss function has a fundamental bug.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import defaultdict


class SimpleContrastiveLoss(nn.Module):
    """
    Dead-simple contrastive loss for testing.
    No hot patches, no complexity, just InfoNCE.
    """

    def __init__(self, temperature=0.2, learnable_temp=True):
        super().__init__()
        if learnable_temp:
            # CLIP-style learnable temperature
            self.log_temperature = nn.Parameter(
                torch.log(torch.tensor(1.0 / temperature))
            )
        else:
            self.register_buffer(
                "log_temperature", torch.log(torch.tensor(1.0 / temperature))
            )

    @property
    def temperature(self):
        return torch.exp(self.log_temperature)

    def forward(self, vision_features, text_features, match_ids):
        """
        Args:
            vision_features: [B, D] unnormalized
            text_features: [B, D] unnormalized
            match_ids: List[str] of length B
        """
        batch_size = vision_features.shape[0]

        # Normalize
        vision_features = F.normalize(vision_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)

        # Compute similarity matrix
        similarity = torch.matmul(vision_features, text_features.T)
        similarity = similarity * torch.exp(self.log_temperature)

        # Build targets for cross-entropy
        # For each row i, the target is any j where match_ids[i] == match_ids[j]

        # Simple approach: for each sample, pick one positive randomly
        targets = []
        for i in range(batch_size):
            # Find all matching indices
            matches = [j for j in range(batch_size) if match_ids[i] == match_ids[j]]

            if not matches:
                # Fallback (shouldn't happen with proper data)
                targets.append(i)
            else:
                # Randomly pick one matching target
                import random

                targets.append(random.choice(matches))

        targets = torch.tensor(targets, dtype=torch.long, device=similarity.device)

        # Vision-to-text loss
        loss_v2t = F.cross_entropy(similarity, targets, reduction="mean")

        # Text-to-vision loss (transpose)
        loss_t2v = F.cross_entropy(similarity.T, targets, reduction="mean")

        # Total loss
        loss = (loss_v2t + loss_t2v) / 2

        return loss


def test_contrastive_learning():
    """
    Test if contrastive learning works with synthetic data.
    """
    print("=" * 80)
    print("MINIMAL CONTRASTIVE LEARNING TEST")
    print("=" * 80)

    # Simulate Flickr30k: 3 images with 5 captions each
    batch_size = 15
    vision_dim = 768
    text_dim = 768

    # Create learnable features (simulating what the model would learn)
    vision_features = nn.Parameter(torch.randn(batch_size, vision_dim) * 0.1)
    text_features = nn.Parameter(torch.randn(batch_size, text_dim) * 0.1)

    # Match IDs: 3 images with 5 captions each
    match_ids = (
        ["img1"] * 5 + ["img2"] * 5 + ["img3"] * 5
    )  # Proper semantic grouping

    print(f"\nBatch size: {batch_size}")
    print(f"Number of unique images: {len(set(match_ids))}")
    print(f"Captions per image: {batch_size // len(set(match_ids))}")
    print(f"Match IDs: {match_ids}")

    # Initialize loss
    loss_fn = SimpleContrastiveLoss(temperature=0.2, learnable_temp=True)

    # Optimizer
    optimizer = torch.optim.Adam(
        [vision_features, text_features] + list(loss_fn.parameters()), lr=0.01
    )

    # Track metrics
    history = defaultdict(list)

    print("\nStarting training loop...")
    print("-" * 80)

    for step in range(200):
        optimizer.zero_grad()

        # Forward pass
        loss = loss_fn(vision_features, text_features, match_ids)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Compute diagnostics (no grad)
        with torch.no_grad():
            v_norm = F.normalize(vision_features, p=2, dim=1)
            t_norm = F.normalize(text_features, p=2, dim=1)
            similarity = torch.matmul(v_norm, t_norm.T)

            # Diagonal similarity (matched pairs)
            diag_sim = torch.diagonal(similarity).mean().item()

            # Off-diagonal similarity (unmatched pairs)
            mask = ~torch.eye(batch_size, dtype=torch.bool, device=similarity.device)
            off_diag_sim = similarity[mask].mean().item()

            # Gap
            gap = diag_sim - off_diag_sim

            # Temperature
            temp = loss_fn.temperature.item()

            # Feature variance
            vision_var = v_norm.var(dim=0).mean().item()
            text_var = t_norm.var(dim=0).mean().item()

        # Log metrics
        history["step"].append(step)
        history["loss"].append(loss.item())
        history["diag_sim"].append(diag_sim)
        history["off_diag_sim"].append(off_diag_sim)
        history["gap"].append(gap)
        history["temperature"].append(temp)
        history["vision_var"].append(vision_var)
        history["text_var"].append(text_var)

        # Print every 20 steps
        if step % 20 == 0:
            print(
                f"Step {step:3d} | Loss: {loss.item():.4f} | Diag: {diag_sim:.4f} | OffDiag: {off_diag_sim:.4f} | Gap: {gap:.4f} | Temp: {temp:.4f}"
            )

    print("-" * 80)
    print("\nFinal Results:")
    print(f"  Loss:            {history['loss'][-1]:.4f} (started at {history['loss'][0]:.4f})")
    print(f"  Diagonal sim:    {history['diag_sim'][-1]:.4f} (started at {history['diag_sim'][0]:.4f})")
    print(f"  Off-diag sim:    {history['off_diag_sim'][-1]:.4f} (started at {history['off_diag_sim'][0]:.4f})")
    print(f"  Gap:             {history['gap'][-1]:.4f} (started at {history['gap'][0]:.4f})")
    print(f"  Temperature:     {history['temperature'][-1]:.4f} (started at {history['temperature'][0]:.4f})")
    print(f"  Vision variance: {history['vision_var'][-1]:.4f}")
    print(f"  Text variance:   {history['text_var'][-1]:.4f}")

    # Evaluation
    print("\n" + "=" * 80)
    print("EVALUATION")
    print("=" * 80)

    final_gap = history["gap"][-1]
    final_diag = history["diag_sim"][-1]
    loss_decreased = history["loss"][-1] < history["loss"][0]

    success = True

    # Check 1: Loss decreased
    if loss_decreased:
        print("✅ Loss decreased - model is learning")
    else:
        print("🔴 Loss did NOT decrease - learning failed")
        success = False

    # Check 2: Diagonal similarity increased
    if history["diag_sim"][-1] > history["diag_sim"][0] + 0.2:
        print("✅ Diagonal similarity increased - positive pairs are learning")
    else:
        print("🔴 Diagonal similarity did not increase enough - positive pairs not learning")
        success = False

    # Check 3: Gap is substantial
    if final_gap > 0.3:
        print(f"✅ Gap is {final_gap:.4f} > 0.3 - good separation between positives and negatives")
    elif final_gap > 0.1:
        print(f"🟡 Gap is {final_gap:.4f} - weak separation (should be > 0.3)")
        success = False
    else:
        print(f"🔴 Gap is {final_gap:.4f} - NO separation - FEATURE COLLAPSE")
        success = False

    # Check 4: Features have variance
    if history["vision_var"][-1] > 0.1 and history["text_var"][-1] > 0.1:
        print("✅ Features have sufficient variance - no dimensional collapse")
    else:
        print("🔴 Features have low variance - dimensional collapse detected")
        success = False

    # Final verdict
    print("\n" + "=" * 80)
    if success:
        print("🎉 SUCCESS! Contrastive learning is working correctly!")
        print("Your loss function and training loop are sound.")
        print("\nIf your full training still fails, the issue is likely:")
        print("  1. Match IDs in your real dataloader")
        print("  2. Pretrained model freezing/gradients")
        print("  3. Learning rate or optimization settings")
    else:
        print("🔴 FAILURE! Contrastive learning is NOT working!")
        print("There is a fundamental issue with your loss function or setup.")
        print("\nDebugging steps:")
        print("  1. Check if match_ids have proper duplicates")
        print("  2. Verify temperature is not too low")
        print("  3. Ensure features are being normalized correctly")
        print("  4. Check for gradient flow issues")

    # Plot results
    print("\n" + "=" * 80)
    print("Generating plots...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Loss over time
    axes[0, 0].plot(history["step"], history["loss"], label="Loss")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    # Plot 2: Similarity over time
    axes[0, 1].plot(history["step"], history["diag_sim"], label="Diagonal Similarity", color="blue")
    axes[0, 1].plot(history["step"], history["off_diag_sim"], label="Off-Diagonal Similarity", color="red", linestyle="--")
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("Similarity")
    axes[0, 1].set_title("Similarity Dynamics")
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    # Plot 3: Gap over time
    axes[1, 0].plot(history["step"], history["gap"], label="Gap (Diag - OffDiag)", color="green")
    axes[1, 0].axhline(y=0.3, color="orange", linestyle="--", label="Target Gap (0.3)")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_ylabel("Gap")
    axes[1, 0].set_title("Separation Gap")
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    # Plot 4: Temperature over time
    axes[1, 1].plot(history["step"], history["temperature"], label="Temperature", color="purple")
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].set_ylabel("Temperature")
    axes[1, 1].set_title("Learned Temperature")
    axes[1, 1].grid(True)
    axes[1, 1].legend()

    plt.tight_layout()
    output_path = "minimal_training_test_results.png"
    plt.savefig(output_path)
    print(f"Plots saved to: {output_path}")

    # Also print final similarity matrix for inspection
    print("\n" + "=" * 80)
    print("Final Similarity Matrix (first 5x5):")
    print("-" * 80)
    with torch.no_grad():
        v_norm = F.normalize(vision_features, p=2, dim=1)
        t_norm = F.normalize(text_features, p=2, dim=1)
        final_sim = torch.matmul(v_norm, t_norm.T)

        # Print first 5x5
        print("        ", end="")
        for j in range(min(5, batch_size)):
            print(f"  {match_ids[j]:6s}", end="")
        print()

        for i in range(min(5, batch_size)):
            print(f"{match_ids[i]:6s}  ", end="")
            for j in range(min(5, batch_size)):
                sim_val = final_sim[i, j].item()
                # Highlight diagonal
                if i == j:
                    print(f"[{sim_val:5.2f}]", end=" ")
                else:
                    print(f" {sim_val:5.2f} ", end=" ")
            print()

    print("\nNote: Diagonal values [in brackets] should be higher than off-diagonal values")

    return success


if __name__ == "__main__":
    success = test_contrastive_learning()

    if success:
        print("\n✅ Test passed! Your loss function works correctly.")
        print("If full training still fails, debug your dataloader and match_ids.")
    else:
        print("\n🔴 Test failed! Fix your loss function before proceeding.")

    print("\n" + "=" * 80)
    print("Next steps:")
    print("1. If test passed: Run diagnose_match_ids.py on your real dataloader")
    print("2. If test failed: Review the loss function implementation")
    print("3. After both pass: Try full training with the fixes")
    print("=" * 80)
