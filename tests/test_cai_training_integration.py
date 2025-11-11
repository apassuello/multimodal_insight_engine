"""
Integration tests for Constitutional AI training pipelines.

Tests the complete training workflows including:
- Phase 1: Critique → Revision → Supervised Fine-tuning
- Phase 2: Preference Generation → Reward Model → PPO
- End-to-end Phase 1 → Phase 2 pipeline
- Checkpoint management and resumption
- Metrics tracking and validation
"""

import pytest
import torch
import torch.nn as nn
from typing import List, Dict, Any
import tempfile
import os
from pathlib import Path

# Import CAI components
from src.safety.constitutional import (
    ConstitutionalPipeline,
    ConstitutionalFramework,
    ConstitutionalPrinciple,
    RLAIFTrainer,
    PPOTrainer,
    RewardModel,
    setup_default_framework
)


class MockConfig:
    """Picklable config for MockLanguageModel."""

    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size


class MockOutput:
    """Picklable output for MockLanguageModel."""

    def __init__(self, logits, loss=None, hidden_states=None):
        self.logits = logits
        self.loss = loss
        self.hidden_states = hidden_states if hidden_states is not None else []


class MockLanguageModel(nn.Module):
    """Mock language model for testing."""

    def __init__(self, vocab_size: int = 1000, hidden_size: int = 256):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=4,
            dim_feedforward=512,
            batch_first=True
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        self.config = MockConfig(hidden_size)

    def forward(self, input_ids, **kwargs):
        embeds = self.embedding(input_ids)
        hidden = self.transformer(embeds)
        logits = self.lm_head(hidden)

        # Return with loss if labels provided
        if 'labels' in kwargs:
            labels = kwargs['labels']
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
            return MockOutput(logits=logits, loss=loss, hidden_states=[hidden])

        return MockOutput(logits=logits, loss=None, hidden_states=[hidden])

    def generate(self, input_ids, **kwargs):
        """Mock generation method."""
        # Simple generation: just return input + one random token
        batch_size = input_ids.size(0)
        new_token = torch.randint(0, self.vocab_size, (batch_size, 1))
        return torch.cat([input_ids, new_token], dim=1)


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2

    def __call__(self, text, **kwargs):
        """Tokenize text."""
        if isinstance(text, str):
            # Simple character-level tokenization
            tokens = [ord(c) % self.vocab_size for c in text[:50]]
            # Ensure minimum length
            if len(tokens) == 0:
                tokens = [self.bos_token_id]

            tokens = tokens[:kwargs.get('max_length', 50)]

            # Pad if needed
            if kwargs.get('padding', False):
                max_len = kwargs.get('max_length', 50)
                tokens = tokens + [self.pad_token_id] * (max_len - len(tokens))

            if kwargs.get('return_tensors') == 'pt':
                return {
                    'input_ids': torch.tensor([tokens]),
                    'attention_mask': torch.tensor([[1] * len(tokens)])
                }
            return {'input_ids': tokens}
        elif isinstance(text, list):
            # Batch tokenization
            batch_tokens = []
            for t in text:
                tokens = [ord(c) % self.vocab_size for c in t[:50]]
                # Ensure minimum length
                if len(tokens) == 0:
                    tokens = [self.bos_token_id]
                batch_tokens.append(tokens)

            if kwargs.get('return_tensors') == 'pt':
                # Pad to same length
                max_len = max(len(t) for t in batch_tokens) if batch_tokens else 1
                padded = [
                    t + [self.pad_token_id] * (max_len - len(t))
                    for t in batch_tokens
                ]
                return {
                    'input_ids': torch.tensor(padded),
                    'attention_mask': torch.tensor([[1] * len(t) + [0] * (max_len - len(t)) for t in batch_tokens])
                }
            return {'input_ids': batch_tokens}

    def decode(self, token_ids, skip_special_tokens=True):
        """Decode tokens to text."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        # Filter special tokens
        if skip_special_tokens:
            token_ids = [
                t for t in token_ids
                if t not in [self.pad_token_id, self.eos_token_id, self.bos_token_id]
            ]

        # Convert to characters
        chars = [chr(t) if 32 <= t < 127 else '?' for t in token_ids]
        return ''.join(chars)


@pytest.fixture
def mock_model():
    """Create a mock language model."""
    model = MockLanguageModel(vocab_size=1000, hidden_size=256)
    model.tokenizer = MockTokenizer(vocab_size=1000)
    return model


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    return MockTokenizer(vocab_size=1000)


@pytest.fixture
def constitutional_framework():
    """Create a constitutional framework for testing."""
    return setup_default_framework()


@pytest.fixture
def test_prompts():
    """Create test prompts for training."""
    return [
        "Explain quantum computing.",
        "What is machine learning?",
        "How do neural networks work?",
        "Describe natural language processing."
    ]


@pytest.fixture
def validation_prompts():
    """Create validation prompts."""
    return [
        "What is artificial intelligence?",
        "Explain deep learning."
    ]


class TestPhase1Training:
    """Tests for Phase 1: Critique-Revision-SFT pipeline."""

    def test_phase1_pipeline_runs(
        self,
        mock_model,
        mock_tokenizer,
        constitutional_framework,
        test_prompts
    ):
        """Test that Phase 1 pipeline completes without errors."""
        pipeline = ConstitutionalPipeline(
            base_model=mock_model,
            tokenizer=mock_tokenizer,
            constitutional_framework=constitutional_framework
        )

        # Run Phase 1 only
        results = pipeline._run_phase1(
            prompts=test_prompts[:2],  # Use small subset
            num_epochs=1,
            num_revisions=1,
            batch_size=2,
            validation_prompts=None
        )

        # Verify results structure
        assert "training_data_size" in results
        assert "sft_results" in results
        assert results["training_data_size"] > 0

    def test_phase1_checkpoint_save_load(
        self,
        mock_model,
        mock_tokenizer,
        constitutional_framework,
        test_prompts
    ):
        """Test Phase 1 checkpoint save and load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = ConstitutionalPipeline(
                base_model=mock_model,
                tokenizer=mock_tokenizer,
                constitutional_framework=constitutional_framework
            )

            # Run Phase 1
            pipeline._run_phase1(
                prompts=test_prompts[:2],
                num_epochs=1,
                num_revisions=1,
                batch_size=2
            )

            # Save checkpoint
            checkpoint_path = os.path.join(tmpdir, "phase1_checkpoint.pt")
            pipeline._save_phase1_checkpoint(checkpoint_path)
            assert os.path.exists(checkpoint_path)

            # Create new pipeline and load checkpoint
            new_pipeline = ConstitutionalPipeline(
                base_model=MockLanguageModel(vocab_size=1000, hidden_size=256),
                tokenizer=mock_tokenizer,
                constitutional_framework=constitutional_framework
            )
            new_pipeline._load_phase1_checkpoint(checkpoint_path)

            # Verify checkpoint was loaded
            assert new_pipeline.phase1_complete


class TestPhase2Training:
    """Tests for Phase 2: Preference-Reward-PPO pipeline."""

    def test_phase2_pipeline_runs(
        self,
        mock_model,
        mock_tokenizer,
        constitutional_framework,
        test_prompts
    ):
        """Test that Phase 2 pipeline completes without errors."""
        pipeline = ConstitutionalPipeline(
            base_model=mock_model,
            tokenizer=mock_tokenizer,
            constitutional_framework=constitutional_framework
        )

        # Mark Phase 1 as complete
        pipeline.phase1_complete = True

        # Run Phase 2 with minimal steps
        results = pipeline._run_phase2(
            prompts=test_prompts[:2],
            num_epochs=1,
            responses_per_prompt=2,
            reward_model_epochs=1,
            ppo_steps=2,
            ppo_batch_size=2,
            ppo_epochs_per_batch=1,
            validation_prompts=None
        )

        # Verify results structure
        assert "preference_pairs" in results
        assert "reward_model_results" in results
        assert "ppo_results" in results
        assert results["preference_pairs"] > 0

    def test_reward_model_training(
        self,
        mock_model,
        mock_tokenizer,
        constitutional_framework
    ):
        """Test reward model training in isolation."""
        # Create reward model
        hidden_size = 256
        reward_model = RewardModel(base_model=mock_model, hidden_size=hidden_size)

        # Create mock preference data
        preference_data = [
            {
                "prompt": "Test prompt 1",
                "chosen": "Good response",
                "rejected": "Bad response",
                "chosen_score": 0.8,
                "rejected_score": 0.2
            },
            {
                "prompt": "Test prompt 2",
                "chosen": "Better response",
                "rejected": "Worse response",
                "chosen_score": 0.9,
                "rejected_score": 0.1
            }
        ]

        # Train reward model
        from src.safety.constitutional import RewardModelTrainer

        trainer = RewardModelTrainer(
            reward_model=reward_model,
            tokenizer=mock_tokenizer,
            learning_rate=1e-4
        )

        results = trainer.train(
            preference_data=preference_data,
            num_epochs=1,
            batch_size=2
        )

        # Verify training completed
        assert "final_loss" in results
        assert "final_accuracy" in results


class TestEndToEndPipeline:
    """Tests for complete end-to-end Phase 1 → Phase 2 pipeline."""

    def test_full_pipeline_runs(
        self,
        mock_model,
        mock_tokenizer,
        constitutional_framework,
        test_prompts,
        validation_prompts
    ):
        """Test complete end-to-end pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = ConstitutionalPipeline(
                base_model=mock_model,
                tokenizer=mock_tokenizer,
                constitutional_framework=constitutional_framework
            )

            # Run full pipeline with minimal parameters
            results = pipeline.train(
                training_prompts=test_prompts[:2],
                phase1_epochs=1,
                phase1_num_revisions=1,
                phase1_batch_size=2,
                phase2_epochs=1,
                phase2_responses_per_prompt=2,
                phase2_reward_model_epochs=1,
                phase2_ppo_steps=2,
                phase2_ppo_batch_size=2,
                phase2_ppo_epochs_per_batch=1,
                validation_prompts=None,
                save_dir=tmpdir
            )

            # Verify both phases completed
            assert results["phase1_complete"]
            assert results["phase2_complete"]
            assert "training_history" in results
            assert "statistics" in results

            # Verify checkpoints were saved
            assert os.path.exists(os.path.join(tmpdir, "phase1_checkpoint.pt"))
            assert os.path.exists(os.path.join(tmpdir, "phase2_checkpoint.pt"))

    def test_pipeline_resume_from_phase1(
        self,
        mock_model,
        mock_tokenizer,
        constitutional_framework,
        test_prompts
    ):
        """Test resuming pipeline from Phase 1 checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First run: Complete Phase 1
            pipeline1 = ConstitutionalPipeline(
                base_model=mock_model,
                tokenizer=mock_tokenizer,
                constitutional_framework=constitutional_framework
            )

            pipeline1._run_phase1(
                prompts=test_prompts[:2],
                num_epochs=1,
                num_revisions=1,
                batch_size=2
            )

            # Save Phase 1 checkpoint
            checkpoint_path = os.path.join(tmpdir, "phase1_checkpoint.pt")
            pipeline1._save_phase1_checkpoint(checkpoint_path)

            # Second run: Resume from Phase 1
            pipeline2 = ConstitutionalPipeline(
                base_model=MockLanguageModel(vocab_size=1000, hidden_size=256),
                tokenizer=mock_tokenizer,
                constitutional_framework=constitutional_framework
            )

            results = pipeline2.train(
                training_prompts=test_prompts[:2],
                phase2_ppo_steps=2,
                phase2_ppo_batch_size=2,
                save_dir=tmpdir,
                resume_from_phase1=True
            )

            # Verify resumed successfully
            assert results["phase1_complete"]
            assert results["phase2_complete"]


class TestRLAIFTrainerIntegration:
    """Tests for RLAIFTrainer integration with PPOTrainer."""

    def test_rlaif_trainer_uses_ppo(
        self,
        mock_model,
        mock_tokenizer,
        constitutional_framework,
        test_prompts
    ):
        """Test that RLAIFTrainer properly delegates to PPOTrainer."""
        # Create reward model
        reward_model = RewardModel(base_model=mock_model, hidden_size=256)

        # Create RLAIF trainer
        rlaif_trainer = RLAIFTrainer(
            policy_model=mock_model,
            constitutional_framework=constitutional_framework,
            reward_model=reward_model
        )

        # Train with minimal parameters
        results = rlaif_trainer.train(
            prompts=test_prompts[:2],
            num_steps=2,
            batch_size=2,
            num_epochs_per_batch=1,
            tokenizer=mock_tokenizer
        )

        # Verify PPO trainer was used
        assert rlaif_trainer.ppo_trainer is not None
        assert "ppo_results" in results
        assert "final_stats" in results

    def test_rlaif_statistics_tracking(
        self,
        mock_model,
        mock_tokenizer,
        constitutional_framework,
        test_prompts
    ):
        """Test that RLAIF trainer tracks statistics correctly."""
        reward_model = RewardModel(base_model=mock_model, hidden_size=256)

        rlaif_trainer = RLAIFTrainer(
            policy_model=mock_model,
            constitutional_framework=constitutional_framework,
            reward_model=reward_model
        )

        results = rlaif_trainer.train(
            prompts=test_prompts[:2],
            num_steps=2,
            batch_size=2,
            tokenizer=mock_tokenizer
        )

        # Verify statistics are tracked
        stats = rlaif_trainer.get_statistics()
        assert "training_iterations" in stats
        assert "total_prompts_processed" in stats
        assert stats["training_iterations"] > 0
        assert stats["total_prompts_processed"] > 0


class TestConstitutionalTrainerIntegration:
    """Tests for ConstitutionalTrainer constitutional loss computation."""

    def test_constitutional_loss_computation(self):
        """Test that constitutional loss is computed during training."""
        from src.training.trainers.constitutional_trainer import ConstitutionalTrainer

        # Create mock components
        model = MockLanguageModel(vocab_size=1000, hidden_size=256)
        model.tokenizer = MockTokenizer(vocab_size=1000)

        # Create mock dataloader
        batch = {
            'input_ids': torch.randint(0, 1000, (2, 50)),
            'labels': torch.randint(0, 1000, (2, 50))
        }

        class MockDataLoader:
            def __init__(self, batch):
                self.batch = batch

            def __iter__(self):
                yield self.batch

            def __len__(self):
                return 1

        train_dataloader = MockDataLoader(batch)

        # Create constitutional trainer with RLAIF
        trainer = ConstitutionalTrainer(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=None,
            use_rlaif=True,
            constitutional_weight=0.5
        )

        # Perform training step
        metrics = trainer.train_step(batch, step=10)  # Step 10 to trigger constitutional loss

        # Verify metrics
        assert "total_loss" in metrics
        assert "lm_loss" in metrics
        assert "constitutional_loss" in metrics

    def test_helper_methods_exist(self):
        """Test that helper methods for constitutional loss exist."""
        from src.training.trainers.constitutional_trainer import ConstitutionalTrainer

        model = MockLanguageModel(vocab_size=1000, hidden_size=256)
        model.tokenizer = MockTokenizer(vocab_size=1000)

        class MockDataLoader:
            def __iter__(self):
                return iter([])

            def __len__(self):
                return 0

        trainer = ConstitutionalTrainer(
            model=model,
            train_dataloader=MockDataLoader(),
            use_rlaif=True
        )

        # Verify helper methods exist
        assert hasattr(trainer, '_compute_constitutional_loss')
        assert hasattr(trainer, '_extract_prompts_from_batch')
        assert hasattr(trainer, '_generate_response_for_evaluation')


class TestMetricsTracking:
    """Tests for metrics tracking throughout training."""

    def test_pipeline_tracks_statistics(
        self,
        mock_model,
        mock_tokenizer,
        constitutional_framework,
        test_prompts
    ):
        """Test that pipeline tracks statistics correctly."""
        pipeline = ConstitutionalPipeline(
            base_model=mock_model,
            tokenizer=mock_tokenizer,
            constitutional_framework=constitutional_framework
        )

        # Run full pipeline
        results = pipeline.train(
            training_prompts=test_prompts[:2],
            phase1_epochs=1,
            phase1_num_revisions=1,
            phase2_ppo_steps=2,
            phase2_ppo_batch_size=2
        )

        # Verify statistics
        stats = pipeline.get_statistics()
        assert "pipeline_stats" in stats
        assert "phase1_complete" in stats
        assert "phase2_complete" in stats
        assert stats["phase1_complete"]
        assert stats["phase2_complete"]

    def test_evaluation_metrics(
        self,
        mock_model,
        mock_tokenizer,
        constitutional_framework,
        validation_prompts
    ):
        """Test constitutional compliance evaluation."""
        pipeline = ConstitutionalPipeline(
            base_model=mock_model,
            tokenizer=mock_tokenizer,
            constitutional_framework=constitutional_framework
        )

        # Evaluate compliance
        eval_results = pipeline.evaluate_constitutional_compliance(
            test_prompts=validation_prompts,
            model=mock_model
        )

        # Verify evaluation metrics
        assert "avg_score" in eval_results
        assert "violation_rate" in eval_results
        assert "num_violations" in eval_results
        assert "total_evaluated" in eval_results
        assert eval_results["total_evaluated"] == len(validation_prompts)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
