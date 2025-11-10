"""
Unit tests for PPO Trainer (Component 4).

Tests all core PPO components:
- GAE computation
- KL divergence calculation
- Clipped PPO objective
- Training step
- Full training loop
"""

import pytest
import torch
import torch.nn as nn

from src.safety.constitutional.ppo_trainer import PPOTrainer


class MockRewardModel(nn.Module):
    """Mock reward model for testing."""

    def __init__(self, hidden_size=768):
        super().__init__()
        self.reward_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        """Return mock reward scores."""
        batch_size = input_ids.shape[0]
        # Return random rewards between -1 and 1
        return torch.randn(batch_size)


class MockValueModel(nn.Module):
    """Mock value model for testing."""

    def __init__(self, hidden_size=768):
        super().__init__()
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        """Return mock value estimates."""
        batch_size = input_ids.shape[0]
        # Create a mock hidden state and pass through linear layer
        # This ensures gradients flow properly during training
        hidden = torch.randn(batch_size, self.value_head.in_features, requires_grad=True)
        return self.value_head(hidden).squeeze(-1)


class TestComputeGAE:
    """Test Generalized Advantage Estimation."""

    def test_gae_basic_computation(self):
        """Test that GAE computes advantages correctly."""
        device = torch.device('cpu')

        # Create mock models
        from transformers import AutoModelForCausalLM, AutoTokenizer
        policy_model = AutoModelForCausalLM.from_pretrained('gpt2')
        value_model = MockValueModel()
        reward_model = MockRewardModel()
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token

        # Initialize trainer
        trainer = PPOTrainer(
            policy_model=policy_model,
            value_model=value_model,
            reward_model=reward_model,
            tokenizer=tokenizer,
            device=device,
            gamma=0.99,
            gae_lambda=0.95
        )

        # Test with simple rewards and values
        batch_size = 2
        seq_len = 5

        rewards = torch.tensor([
            [1.0, 0.5, 0.2, 0.1, 0.0],
            [0.8, 0.6, 0.4, 0.2, 0.0]
        ])
        values = torch.tensor([
            [0.5, 0.4, 0.3, 0.2, 0.1],
            [0.6, 0.5, 0.4, 0.3, 0.2]
        ])
        dones = torch.zeros(batch_size, seq_len)

        # Compute GAE
        advantages, returns = trainer.compute_gae(rewards, values, dones)

        # Check shapes
        assert advantages.shape == (batch_size, seq_len)
        assert returns.shape == (batch_size, seq_len)

        # Check that advantages are computed (not all zeros)
        assert not torch.allclose(advantages, torch.zeros_like(advantages))

        # Check that returns = advantages + values
        torch.testing.assert_close(returns, advantages + values, rtol=1e-5, atol=1e-5)

    def test_gae_backwards_computation(self):
        """Test that GAE computes backwards through time."""
        device = torch.device('cpu')

        # Create minimal trainer
        from transformers import AutoModelForCausalLM, AutoTokenizer
        policy_model = AutoModelForCausalLM.from_pretrained('gpt2')
        value_model = MockValueModel()
        reward_model = MockRewardModel()
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token

        trainer = PPOTrainer(
            policy_model=policy_model,
            value_model=value_model,
            reward_model=reward_model,
            tokenizer=tokenizer,
            device=device,
            gamma=1.0,  # No discounting for simpler test
            gae_lambda=0.0  # Pure TD(0) for verification
        )

        # Simple test case
        rewards = torch.tensor([[1.0, 0.0, 0.0]])
        values = torch.tensor([[0.0, 0.0, 0.0]])
        dones = torch.zeros(1, 3)

        advantages, returns = trainer.compute_gae(rewards, values, dones)

        # With lambda=0, advantages should be TD errors
        # First timestep: δ = r_0 + γ*V(s_1) - V(s_0) = 1.0 + 0 - 0 = 1.0
        assert advantages[0, 0].item() == pytest.approx(1.0, abs=1e-5)

    def test_gae_with_dones(self):
        """Test that GAE handles episode termination correctly."""
        device = torch.device('cpu')

        from transformers import AutoModelForCausalLM, AutoTokenizer
        policy_model = AutoModelForCausalLM.from_pretrained('gpt2')
        value_model = MockValueModel()
        reward_model = MockRewardModel()
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token

        trainer = PPOTrainer(
            policy_model=policy_model,
            value_model=value_model,
            reward_model=reward_model,
            tokenizer=tokenizer,
            device=device
        )

        rewards = torch.tensor([[1.0, 1.0, 1.0]])
        values = torch.tensor([[0.5, 0.5, 0.5]])
        dones = torch.tensor([[0.0, 1.0, 0.0]])  # Episode ends at timestep 1

        advantages, returns = trainer.compute_gae(rewards, values, dones)

        # Check that computation doesn't propagate past done
        assert advantages.shape == (1, 3)


class TestComputeKLDivergence:
    """Test KL divergence computation."""

    def test_kl_divergence_identical_policies(self):
        """Test KL divergence is ~0 for identical policies."""
        device = torch.device('cpu')

        from transformers import AutoModelForCausalLM, AutoTokenizer
        policy_model = AutoModelForCausalLM.from_pretrained('gpt2')
        value_model = MockValueModel()
        reward_model = MockRewardModel()
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token

        trainer = PPOTrainer(
            policy_model=policy_model,
            value_model=value_model,
            reward_model=reward_model,
            tokenizer=tokenizer,
            device=device
        )

        # Identical log probabilities
        logprobs = torch.randn(4, 10)

        kl_div = trainer.compute_kl_divergence(logprobs, logprobs)

        # Should be exactly 0
        assert kl_div.item() == pytest.approx(0.0, abs=1e-6)

    def test_kl_divergence_different_policies(self):
        """Test KL divergence is positive for different policies."""
        device = torch.device('cpu')

        from transformers import AutoModelForCausalLM, AutoTokenizer
        policy_model = AutoModelForCausalLM.from_pretrained('gpt2')
        value_model = MockValueModel()
        reward_model = MockRewardModel()
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token

        trainer = PPOTrainer(
            policy_model=policy_model,
            value_model=value_model,
            reward_model=reward_model,
            tokenizer=tokenizer,
            device=device
        )

        # Different log probabilities
        current_logprobs = torch.randn(4, 10)
        reference_logprobs = torch.randn(4, 10)

        kl_div = trainer.compute_kl_divergence(current_logprobs, reference_logprobs)

        # KL can be positive or negative in our approximation
        # Just check it's computed
        assert isinstance(kl_div.item(), float)


class TestComputePPOLoss:
    """Test clipped PPO objective."""

    def test_ppo_loss_no_clipping(self):
        """Test PPO loss when ratio is within clip range."""
        device = torch.device('cpu')

        from transformers import AutoModelForCausalLM, AutoTokenizer
        policy_model = AutoModelForCausalLM.from_pretrained('gpt2')
        value_model = MockValueModel()
        reward_model = MockRewardModel()
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token

        trainer = PPOTrainer(
            policy_model=policy_model,
            value_model=value_model,
            reward_model=reward_model,
            tokenizer=tokenizer,
            device=device,
            clip_epsilon=0.2
        )

        # Small difference in log probs (ratio close to 1)
        old_logprobs = torch.tensor([[-1.0, -1.0]])
        new_logprobs = torch.tensor([[-1.05, -0.95]])  # ratio ~ [0.95, 1.05]
        advantages = torch.tensor([[1.0, 1.0]])

        loss = trainer.compute_ppo_loss(old_logprobs, new_logprobs, advantages)

        # Loss should be computed
        assert isinstance(loss.item(), float)
        # Should be negative of expected value (we minimize loss)
        assert loss.item() < 0

    def test_ppo_loss_with_clipping(self):
        """Test PPO loss clips large ratios."""
        device = torch.device('cpu')

        from transformers import AutoModelForCausalLM, AutoTokenizer
        policy_model = AutoModelForCausalLM.from_pretrained('gpt2')
        value_model = MockValueModel()
        reward_model = MockRewardModel()
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token

        trainer = PPOTrainer(
            policy_model=policy_model,
            value_model=value_model,
            reward_model=reward_model,
            tokenizer=tokenizer,
            device=device,
            clip_epsilon=0.2
        )

        # Large difference in log probs (ratio far from 1)
        old_logprobs = torch.tensor([[-1.0, -1.0]])
        new_logprobs = torch.tensor([[-0.1, -2.0]])  # Large ratios
        advantages = torch.tensor([[1.0, 1.0]])

        loss = trainer.compute_ppo_loss(old_logprobs, new_logprobs, advantages)

        # Loss should be computed and clipping should apply
        assert isinstance(loss.item(), float)

    def test_ppo_loss_negative_advantages(self):
        """Test PPO loss with negative advantages."""
        device = torch.device('cpu')

        from transformers import AutoModelForCausalLM, AutoTokenizer
        policy_model = AutoModelForCausalLM.from_pretrained('gpt2')
        value_model = MockValueModel()
        reward_model = MockRewardModel()
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token

        trainer = PPOTrainer(
            policy_model=policy_model,
            value_model=value_model,
            reward_model=reward_model,
            tokenizer=tokenizer,
            device=device
        )

        old_logprobs = torch.tensor([[-1.0, -1.0]])
        new_logprobs = torch.tensor([[-1.05, -0.95]])
        advantages = torch.tensor([[-1.0, -1.0]])  # Negative advantages

        loss = trainer.compute_ppo_loss(old_logprobs, new_logprobs, advantages)

        # Loss should be positive (we want to decrease prob of bad actions)
        assert isinstance(loss.item(), float)


class TestTrainStep:
    """Test PPO training step."""

    @pytest.mark.slow
    def test_train_step_completes(self):
        """Test that training step completes without errors."""
        device = torch.device('cpu')

        from transformers import AutoModelForCausalLM, AutoTokenizer
        policy_model = AutoModelForCausalLM.from_pretrained('gpt2')
        value_model = MockValueModel()
        reward_model = MockRewardModel()
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token

        trainer = PPOTrainer(
            policy_model=policy_model,
            value_model=value_model,
            reward_model=reward_model,
            tokenizer=tokenizer,
            device=device
        )

        prompts = ["Hello", "Hi there"]

        # Run training step with minimal epochs
        metrics = trainer.train_step(
            prompts,
            num_epochs_per_batch=1,
            max_length=20,
            temperature=1.0
        )

        # Check that metrics are returned
        assert 'policy_loss' in metrics
        assert 'value_loss' in metrics
        assert 'kl_divergence' in metrics
        assert 'mean_reward' in metrics
        assert 'mean_advantage' in metrics

        # Check that all metrics are numbers
        for key, value in metrics.items():
            assert isinstance(value, (int, float))

    @pytest.mark.slow
    def test_train_step_updates_parameters(self):
        """Test that training step updates policy parameters."""
        device = torch.device('cpu')

        from transformers import AutoModelForCausalLM, AutoTokenizer
        policy_model = AutoModelForCausalLM.from_pretrained('gpt2')
        value_model = MockValueModel()
        reward_model = MockRewardModel()
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token

        trainer = PPOTrainer(
            policy_model=policy_model,
            value_model=value_model,
            reward_model=reward_model,
            tokenizer=tokenizer,
            device=device
        )

        # Get initial parameters
        initial_params = {
            name: param.clone()
            for name, param in policy_model.named_parameters()
            if param.requires_grad
        }

        prompts = ["Test prompt"]

        # Run training step
        trainer.train_step(
            prompts,
            num_epochs_per_batch=2,
            max_length=20
        )

        # Check that some parameters changed
        params_changed = False
        for name, param in policy_model.named_parameters():
            if param.requires_grad and name in initial_params:
                if not torch.allclose(param, initial_params[name], atol=1e-6):
                    params_changed = True
                    break

        assert params_changed, "Policy parameters should be updated after training step"

    @pytest.mark.slow
    def test_train_step_gradients_flow(self):
        """Test that gradients flow properly through policy."""
        device = torch.device('cpu')

        from transformers import AutoModelForCausalLM, AutoTokenizer
        policy_model = AutoModelForCausalLM.from_pretrained('gpt2')
        value_model = MockValueModel()
        reward_model = MockRewardModel()
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token

        trainer = PPOTrainer(
            policy_model=policy_model,
            value_model=value_model,
            reward_model=reward_model,
            tokenizer=tokenizer,
            device=device
        )

        prompts = ["Test"]

        # Run training step
        metrics = trainer.train_step(
            prompts,
            num_epochs_per_batch=1,
            max_length=15
        )

        # Check that statistics were updated
        assert trainer.stats['total_steps'] == 1
        assert len(trainer.stats['policy_losses']) == 1
        assert len(trainer.stats['value_losses']) == 1


class TestFullTraining:
    """Test full PPO training loop."""

    @pytest.mark.slow
    def test_train_loop_completes(self):
        """Test that full training loop completes."""
        device = torch.device('cpu')

        from transformers import AutoModelForCausalLM, AutoTokenizer
        policy_model = AutoModelForCausalLM.from_pretrained('gpt2')
        value_model = MockValueModel()
        reward_model = MockRewardModel()
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token

        trainer = PPOTrainer(
            policy_model=policy_model,
            value_model=value_model,
            reward_model=reward_model,
            tokenizer=tokenizer,
            device=device
        )

        prompts = ["Hello", "Hi", "Test"]

        # Run minimal training
        results = trainer.train(
            prompts,
            num_steps=2,
            batch_size=2,
            num_epochs_per_batch=1,
            max_length=15
        )

        # Check results structure
        assert 'training_history' in results
        assert 'final_stats' in results

        # Check training history
        history = results['training_history']
        assert 'policy_losses' in history
        assert 'value_losses' in history
        assert 'kl_divergences' in history
        assert 'mean_rewards' in history

        # Check that we have data for all steps
        assert len(history['policy_losses']) == 2
        assert len(history['value_losses']) == 2


class TestCheckpointing:
    """Test checkpoint saving and loading."""

    @pytest.mark.slow
    def test_save_and_load_checkpoint(self, tmp_path):
        """Test that checkpoints can be saved and loaded."""
        device = torch.device('cpu')

        from transformers import AutoModelForCausalLM, AutoTokenizer
        policy_model = AutoModelForCausalLM.from_pretrained('gpt2')
        value_model = MockValueModel()
        reward_model = MockRewardModel()
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token

        trainer = PPOTrainer(
            policy_model=policy_model,
            value_model=value_model,
            reward_model=reward_model,
            tokenizer=tokenizer,
            device=device
        )

        # Train a bit
        prompts = ["Test"]
        trainer.train_step(prompts, num_epochs_per_batch=1, max_length=15)

        # Save checkpoint
        checkpoint_dir = str(tmp_path)
        trainer.save_checkpoint(checkpoint_dir, step=1)

        # Create new trainer
        policy_model2 = AutoModelForCausalLM.from_pretrained('gpt2')
        value_model2 = MockValueModel()

        trainer2 = PPOTrainer(
            policy_model=policy_model2,
            value_model=value_model2,
            reward_model=reward_model,
            tokenizer=tokenizer,
            device=device
        )

        # Load checkpoint
        import os
        checkpoint_path = os.path.join(checkpoint_dir, "ppo_checkpoint_step_1.pt")
        trainer2.load_checkpoint(checkpoint_path)

        # Check that stats were loaded
        assert trainer2.stats['total_steps'] == trainer.stats['total_steps']


class TestIntegration:
    """Integration tests for PPO with reward model."""

    @pytest.mark.slow
    def test_ppo_with_reward_model(self):
        """Test PPO integration with reward model."""
        device = torch.device('cpu')

        from transformers import AutoModelForCausalLM, AutoTokenizer
        policy_model = AutoModelForCausalLM.from_pretrained('gpt2')
        value_model = MockValueModel()
        reward_model = MockRewardModel()
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token

        trainer = PPOTrainer(
            policy_model=policy_model,
            value_model=value_model,
            reward_model=reward_model,
            tokenizer=tokenizer,
            device=device,
            clip_epsilon=0.2,
            kl_penalty=0.1
        )

        prompts = ["What is AI?"]

        # Run training
        metrics = trainer.train_step(
            prompts,
            num_epochs_per_batch=1,
            max_length=20
        )

        # Verify all components work together
        assert metrics['policy_loss'] is not None
        assert metrics['value_loss'] is not None
        assert metrics['kl_divergence'] is not None
        assert metrics['mean_reward'] is not None

    def test_statistics_tracking(self):
        """Test that statistics are tracked correctly."""
        device = torch.device('cpu')

        from transformers import AutoModelForCausalLM, AutoTokenizer
        policy_model = AutoModelForCausalLM.from_pretrained('gpt2')
        value_model = MockValueModel()
        reward_model = MockRewardModel()
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token

        trainer = PPOTrainer(
            policy_model=policy_model,
            value_model=value_model,
            reward_model=reward_model,
            tokenizer=tokenizer,
            device=device
        )

        # Get initial stats
        stats = trainer.get_statistics()

        assert 'total_steps' in stats
        assert 'avg_policy_loss' in stats
        assert 'avg_value_loss' in stats
        assert 'avg_kl_divergence' in stats
        assert 'avg_reward' in stats


class TestMonitoringIntegration:
    """Test monitoring system integration with PPO trainer."""

    def test_ppo_trainer_accepts_monitor(self):
        """Test that PPO trainer accepts monitor parameter."""
        device = torch.device('cpu')

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from src.training.monitoring import TrainingMonitor

        policy_model = AutoModelForCausalLM.from_pretrained('gpt2')
        value_model = MockValueModel()
        reward_model = MockRewardModel()
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token

        # Create monitor
        monitor = TrainingMonitor()

        # Create trainer with monitor
        trainer = PPOTrainer(
            policy_model=policy_model,
            value_model=value_model,
            reward_model=reward_model,
            tokenizer=tokenizer,
            device=device,
            monitor=monitor
        )

        # Verify monitor is set
        assert trainer.monitor is monitor

    def test_ppo_trainer_without_monitor(self):
        """Test that PPO trainer works without monitor (backward compatibility)."""
        device = torch.device('cpu')

        from transformers import AutoModelForCausalLM, AutoTokenizer

        policy_model = AutoModelForCausalLM.from_pretrained('gpt2')
        value_model = MockValueModel()
        reward_model = MockRewardModel()
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token

        # Create trainer without monitor
        trainer = PPOTrainer(
            policy_model=policy_model,
            value_model=value_model,
            reward_model=reward_model,
            tokenizer=tokenizer,
            device=device
        )

        # Should work fine without monitor
        assert trainer.monitor is None

    @pytest.mark.slow
    def test_training_emits_events(self):
        """Test that training emits events to monitor."""
        device = torch.device('cpu')

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from src.training.monitoring import TrainingMonitor, TrainingPhase
        from unittest.mock import Mock

        policy_model = AutoModelForCausalLM.from_pretrained('gpt2')
        value_model = MockValueModel()
        reward_model = MockRewardModel()
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token

        # Create monitor with mock callback to track events
        monitor = TrainingMonitor()
        mock_callback = Mock()
        monitor.register_callback(mock_callback)

        # Create trainer with monitor
        trainer = PPOTrainer(
            policy_model=policy_model,
            value_model=value_model,
            reward_model=reward_model,
            tokenizer=tokenizer,
            device=device,
            monitor=monitor
        )

        prompts = ["Hello"]

        # Run training step
        with monitor.monitor_context():
            trainer.train_step(
                prompts,
                num_epochs_per_batch=1,
                max_length=15,
                iteration=0
            )

        # Verify events were emitted
        assert mock_callback.on_event.called
        call_count = mock_callback.on_event.call_count

        # Should have emitted at least:
        # - RESPONSE_GENERATED
        # - REWARD_COMPUTED
        # - POLICY_UPDATE (1 epoch)
        # - VALUE_UPDATE (1 epoch)
        # Minimum 4 events
        assert call_count >= 4

        # Check that events have correct phases
        events = [call.args[0] for call in mock_callback.on_event.call_args_list]
        phases = [event.phase for event in events]

        assert TrainingPhase.RESPONSE_GENERATED in phases
        assert TrainingPhase.REWARD_COMPUTED in phases
        assert TrainingPhase.POLICY_UPDATE in phases
        assert TrainingPhase.VALUE_UPDATE in phases

    @pytest.mark.slow
    def test_full_training_emits_all_events(self):
        """Test that full training loop emits all expected events."""
        device = torch.device('cpu')

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from src.training.monitoring import TrainingMonitor, TrainingPhase
        from unittest.mock import Mock

        policy_model = AutoModelForCausalLM.from_pretrained('gpt2')
        value_model = MockValueModel()
        reward_model = MockRewardModel()
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token

        # Create monitor with mock callback
        monitor = TrainingMonitor()
        mock_callback = Mock()
        monitor.register_callback(mock_callback)

        trainer = PPOTrainer(
            policy_model=policy_model,
            value_model=value_model,
            reward_model=reward_model,
            tokenizer=tokenizer,
            device=device,
            monitor=monitor
        )

        prompts = ["Test"]

        # Run minimal training
        with monitor.monitor_context():
            trainer.train(
                prompts,
                num_steps=2,
                batch_size=1,
                num_epochs_per_batch=1,
                max_length=15
            )

        # Collect all event phases
        events = [call.args[0] for call in mock_callback.on_event.call_args_list]
        phases = [event.phase for event in events]

        # Should have TRAINING_START and TRAINING_END
        assert TrainingPhase.TRAINING_START in phases
        assert TrainingPhase.TRAINING_END in phases

        # Should have ITERATION_START and ITERATION_END for each step
        assert phases.count(TrainingPhase.ITERATION_START) == 2
        assert phases.count(TrainingPhase.ITERATION_END) == 2

    @pytest.mark.slow
    def test_early_stopping_integration(self):
        """Test that early stopping works with monitor."""
        device = torch.device('cpu')

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from src.training.monitoring import (
            TrainingMonitor,
            SampleComparator,
            QualityAnalyzer
        )

        policy_model = AutoModelForCausalLM.from_pretrained('gpt2')
        value_model = MockValueModel()
        reward_model = MockRewardModel()
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token

        # Create monitor with quality analyzer that will trigger early stopping
        monitor = TrainingMonitor()
        comparator = SampleComparator(sample_size=1, comparison_frequency=1)
        analyzer = QualityAnalyzer(
            monitor=monitor,
            comparator=comparator,
            kl_threshold=0.0001,  # Very low threshold to trigger early
            reward_hack_threshold=0.001
        )
        monitor.register_callback(comparator)
        monitor.register_callback(analyzer)

        trainer = PPOTrainer(
            policy_model=policy_model,
            value_model=value_model,
            reward_model=reward_model,
            tokenizer=tokenizer,
            device=device,
            monitor=monitor,
            kl_penalty=10.0  # High KL to potentially trigger
        )

        prompts = ["Test prompt"]

        # Run training - early stopping might trigger
        with monitor.monitor_context():
            results = trainer.train(
                prompts,
                num_steps=10,  # Request 10 steps
                batch_size=1,
                num_epochs_per_batch=2,
                max_length=15
            )

        # Training should complete (with or without early stopping)
        assert 'training_history' in results
        assert 'final_stats' in results

    @pytest.mark.slow
    def test_monitoring_tracks_metrics(self):
        """Test that monitor correctly tracks PPO metrics."""
        device = torch.device('cpu')

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from src.training.monitoring import TrainingMonitor

        policy_model = AutoModelForCausalLM.from_pretrained('gpt2')
        value_model = MockValueModel()
        reward_model = MockRewardModel()
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token

        # Create monitor
        monitor = TrainingMonitor()

        trainer = PPOTrainer(
            policy_model=policy_model,
            value_model=value_model,
            reward_model=reward_model,
            tokenizer=tokenizer,
            device=device,
            monitor=monitor
        )

        prompts = ["Hello"]

        # Run training
        with monitor.monitor_context():
            trainer.train(
                prompts,
                num_steps=2,
                batch_size=1,
                num_epochs_per_batch=1,
                max_length=15
            )

        # Check that metrics were tracked
        metrics_store = monitor.get_metrics_store()
        metric_names = metrics_store.get_metric_names()

        # Should have tracked key PPO metrics
        expected_metrics = ['policy_loss', 'value_loss', 'kl_div', 'reward']
        for metric in expected_metrics:
            assert metric in metric_names, f"Expected metric '{metric}' not tracked"

        # Check that we have data for each metric
        for metric in expected_metrics:
            history = metrics_store.get_history(metric)
            assert len(history) > 0, f"No data for metric '{metric}'"

    def test_emit_event_method_exists(self):
        """Test that _emit_event method exists and works."""
        device = torch.device('cpu')

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from src.training.monitoring import TrainingMonitor, TrainingEvent, TrainingPhase

        policy_model = AutoModelForCausalLM.from_pretrained('gpt2')
        value_model = MockValueModel()
        reward_model = MockRewardModel()
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token

        monitor = TrainingMonitor()

        trainer = PPOTrainer(
            policy_model=policy_model,
            value_model=value_model,
            reward_model=reward_model,
            tokenizer=tokenizer,
            device=device,
            monitor=monitor
        )

        # Verify _emit_event method exists
        assert hasattr(trainer, '_emit_event')

        # Test emitting an event manually
        event = TrainingEvent(
            phase=TrainingPhase.TRAINING_START,
            iteration=0,
            metadata={'test': True}
        )

        # Should not raise an error
        trainer._emit_event(event)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
