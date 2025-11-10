# Axis 4: Repository Structure & Modernization

**Timeline**: Weeks 1-16 (Ongoing, parallel with other axes)
**Effort**: 140-200 hours
**Priority**: 🟢 MEDIUM (but continuous)

## Overview

Modernize legacy patterns, improve developer experience, organize documentation, and establish sustainable repository practices. This axis runs in parallel with others.

## Current State

- **Modernization Score**: 5.0/10 (Legacy Patterns)
- **Developer Experience**: 5.5/10 (Friction Points)
- **Documentation**: 6.0/10 (Scattered)
- **Root Files**: 25+ markdown files (navigation chaos)

## Target State (After Week 16)

- **Modernization Score**: 8.0/10 ✅
- **Developer Experience**: 7.5/10 ✅
- **Documentation**: 8.5/10 ✅
- **Root Files**: 5-6 essential files only ✅

---

## Phase 1: Quick Wins (Weeks 1-3, parallel with Axis 1)

### 1. Replace Print Statements with Logging (8-12 hours)

**Problem**: 633 `print()` statements in production code

**Solution**: Centralized logging system

**Implementation**:
```python
# src/utils/logging_config.py
import logging
import sys

def setup_logging(level="INFO"):
    """Setup centralized logging."""
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('training.log')
        ]
    )

# Replace print statements:
# OLD: print(f"Training loss: {loss}")
# NEW: logger.info(f"Training loss: {loss}")
```

**Tasks**:
- [ ] Create `logging_config.py`
- [ ] Replace prints in top 10 most-used files
- [ ] Add logging levels (DEBUG, INFO, WARNING, ERROR)
- [ ] Document logging standards in CLAUDE.md

**See**: `modernization-plan.md` section 1

---

### 2. Organize Root Documentation (6-8 hours)

**Problem**: 25+ markdown files at root

**Solution**: Move to logical subdirectories

**New Structure**:
```
Root (only essentials):
├── README.md
├── GETTING_STARTED.md
├── CLAUDE.md
├── CRITICAL_README.md
└── IMPROVEMENT_PLAN.md

docs/
├── improvement-plan/     (this directory!)
├── audits/               (move audit reports here)
├── assessments/          (move assessment files here)
├── constitutional-ai/    (existing, keep)
├── testing/              (existing, keep)
└── archive/              (legacy docs)
```

**Tasks**:
- [ ] Create `docs/audits/` and `docs/assessments/`
- [ ] Move audit/assessment files from root
- [ ] Update all internal links
- [ ] Add navigation file: `docs/README.md`

**See**: `dx-improvements.md` section 2

---

### 3. Consolidate Demo Scripts (4-6 hours)

**Problem**: 24 demos in `/demos/` with no organization

**Solution**: Categorize and document

**Categories**:
- `demos/language-models/` (5 scripts)
- `demos/translation/` (4 scripts)
- `demos/multimodal/` (6 scripts)
- `demos/safety/` (4 scripts)
- `demos/optimization/` (3 scripts)
- `demos/tutorials/` (2 scripts)

**Tasks**:
- [ ] Create category subdirectories
- [ ] Move demos to categories
- [ ] Create `demos/README.md` with navigation
- [ ] Add difficulty levels (Beginner/Intermediate/Advanced)
- [ ] Add time estimates for each demo

---

## Phase 2: Legacy Code Modernization (Weeks 4-8, parallel with Axis 2)

### 4. Consolidate Dataset Classes (30-40 hours)

**Problem**: 30+ dataset variants with duplication

**Current**:
```
src/data/
├── iwslt_dataset.py
├── europarl_dataset.py
├── wmt_dataset.py
├── wikipedia_dataset.py
├── language_modeling.py
├── sequence_data.py
├── curriculum_dataset.py
├── constitutional_dataset.py
├── combined_dataset.py
├── image_dataset.py
└── multimodal_dataset.py
... (20+ more)
```

**Target**: 10-15 well-designed classes with composition

**Strategy**:
```python
# Base datasets
class TextDataset(BaseDataset):
    """Base class for text datasets."""

class ImageDataset(BaseDataset):
    """Base class for image datasets."""

# Composition for specific datasets
class TranslationDataset(TextDataset):
    """Generic translation dataset."""

    def __init__(self, source, source_lang, target_lang):
        # Load from any source (IWSLT, Europarl, WMT)
        super().__init__(source)
        self.source_lang = source_lang
        self.target_lang = target_lang
```

**Tasks**:
- [ ] Create `BaseDataset` with common functionality
- [ ] Consolidate translation datasets → `TranslationDataset`
- [ ] Consolidate language modeling → `LanguageModelingDataset`
- [ ] Use composition instead of inheritance where possible
- [ ] Deprecate old dataset classes (don't delete immediately)

**Reduction**: 30 classes → 10-15 with <5% duplication

**See**: `modernization-plan.md` section 4

---

### 5. Add Comprehensive Type Hints (20-25 hours)

**Problem**: 15+ files without type hints

**Target**: 100% type hint coverage on public APIs

**Priority Files**:
- `src/data/multimodal_dataset.py` (58% coverage)
- `src/training/trainers/multimodal_trainer.py` (54% coverage)
- All files in `src/utils/` (varies)

**Implementation**:
```python
# Before
def train_epoch(self, dataloader):
    # No type hints
    pass

# After
def train_epoch(
    self,
    dataloader: DataLoader,
    epoch: int
) -> Dict[str, float]:
    """Train for one epoch.

    Args:
        dataloader: Training data loader
        epoch: Current epoch number

    Returns:
        Dictionary of metrics
    """
    pass
```

**Tasks**:
- [ ] Add type hints to all public APIs
- [ ] Run mypy in strict mode
- [ ] Fix mypy errors
- [ ] Add to CI pipeline

---

### 6. Remove Debug Code (12-16 hours)

**Problem**: 16+ debug markers, embedded debug code

**Examples**:
```python
# CRITICAL PATCH: If input_dim == 768 (ViT-base, BERT-base)...
if input_dim == 768:
    projection_dim = 768
    print(f"CRITICAL PATCH: ...")
```

**Solution**: Convert to feature flags or proper configuration

```python
# Configuration-based approach
class ModelConfig:
    input_dim: int = 768
    projection_dim: Optional[int] = None  # Auto-detect if None

    def get_projection_dim(self) -> int:
        """Get projection dimension with auto-detection."""
        if self.projection_dim is not None:
            return self.projection_dim
        # Auto-detect based on input_dim
        return self.input_dim
```

**Tasks**:
- [ ] Identify all "CRITICAL PATCH" markers
- [ ] Convert to configuration
- [ ] Remove "DEBUG:", "TODO:", "HACK:" comments
- [ ] Create proper feature flags where needed

---

## Phase 3: Developer Experience (Weeks 9-12, parallel with Axis 3)

### 7. Improve Installation Process (10-15 hours)

**Current Issues**:
- 331 dependencies, 15-20 min install
- No dependency groups (dev, test, prod)
- No lock file

**Solution**: Dependency management

**Create `pyproject.toml`**:
```toml
[project]
name = "multimodal_insight_engine"
version = "0.1.0"
dependencies = [
    "torch>=2.1.0",
    "transformers>=4.49.0",
    # ... core dependencies only
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3.5",
    "black>=25.1.0",
    "flake8>=7.1.2",
]
test = [
    "pytest>=8.3.5",
    "pytest-cov>=4.0",
]
docs = [
    "sphinx>=7.0.0",
    "sphinx-rtd-theme>=2.0.0",
]
```

**Tasks**:
- [ ] Create `pyproject.toml`
- [ ] Separate dependencies into groups
- [ ] Generate `requirements-dev.txt`, `requirements-test.txt`
- [ ] Add to GETTING_STARTED.md
- [ ] Create lock file (pip-tools or poetry)

---

### 8. Enhanced Error Messages (8-10 hours)

**Problem**: Generic error messages, hard to debug

**Solution**: Informative exceptions

```python
# Before
raise ValueError("Invalid input")

# After
raise ValueError(
    f"Invalid input dimension: expected {expected_dim}, "
    f"got {actual_dim}. "
    f"Hint: Check your model configuration. "
    f"For ViT models, use input_dim=768."
)
```

**Tasks**:
- [ ] Audit top 20 most common errors
- [ ] Add context and hints to error messages
- [ ] Create custom exception classes
- [ ] Add troubleshooting links

---

### 9. Makefile Improvements (4-6 hours)

**Current**: Basic Makefile exists

**Enhancements**:
```makefile
.PHONY: help install test test-fast lint format check clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install all dependencies
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

test:  ## Run all tests with coverage
	pytest --cov=src --cov-report=term-missing --cov-report=html

test-fast:  ## Run fast tests only
	pytest -k "not slow" -n auto

lint:  ## Run linters
	flake8 src/ tests/
	mypy src/

format:  ## Format code
	black src/ tests/
	isort src/ tests/

check: lint test  ## Run all checks

clean:  ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
```

---

## Phase 4: Documentation Excellence (Weeks 13-16)

### 10. Create Missing Core Documentation (40-50 hours)

**Priority Documents**:

**ARCHITECTURE.md** (16-20 hours):
- System overview with diagrams
- Component interactions
- Design patterns used
- Technology choices
- Extension points

**API_REFERENCE.md** (12-16 hours):
- Manual API overview
- Auto-generated Sphinx docs
- Code examples
- Common use cases

**TRAINING_GUIDE.md** (12-16 hours):
- How to train models
- Configuration options
- Hyperparameter tuning
- Troubleshooting
- Example workflows

**See**: `documentation-strategy.md` for templates

---

### 11. Architecture Decision Records (12-16 hours)

**Create `docs/decisions/` with ADRs**:

Example ADR structure:
```markdown
# ADR-001: Use Pydantic for Configuration

## Status
Accepted

## Context
We need a unified configuration system. Current state has 4 different
approaches (dataclasses, argparse, dicts, hardcoded).

## Decision
Use Pydantic for all configuration with:
- Type validation
- Default values
- Environment variable support
- JSON/YAML loading

## Consequences
Positive:
- Type safety
- Better error messages
- Easier testing

Negative:
- Additional dependency
- Migration effort

## Alternatives Considered
- Dataclasses + dacite
- OmegaConf
- Hydra
```

**Priority ADRs**:
- [ ] Configuration system choice
- [ ] Loss function architecture
- [ ] Trainer hierarchy design
- [ ] Test strategy
- [ ] Tokenization approach
- [ ] Multimodal fusion strategy
- [ ] Safety architecture
- [ ] Optimization strategies
- [ ] Dataset consolidation
- [ ] Logging and monitoring

---

### 12. Contributor Guide (6-8 hours)

**Create `CONTRIBUTING.md`**:
- Development setup
- Coding standards
- PR process
- Review checklist
- Testing requirements
- Documentation requirements
- How to add new features

---

## Success Metrics

After completing Axis 4, you should have:

✅ **Print Statements**: 633 → 0 (all use logging)
✅ **Root Files**: 25+ → 5-6
✅ **Dataset Classes**: 30 → 10-15 with clear purpose
✅ **Type Hints**: 70% → 100% on public APIs
✅ **Documentation**: Complete (ARCHITECTURE, API, TRAINING, 10 ADRs)
✅ **Developer Experience**: 5.5/10 → 7.5/10
✅ **Onboarding Time**: 8-12 hours → 2-4 hours

---

## Continuous Improvements

**Ongoing (after Week 16)**:
- Add ADRs for new decisions
- Update documentation with new features
- Refactor as patterns emerge
- Improve error messages based on feedback
- Optimize developer workflow

---

## Risk Mitigation

**Risk**: Too many changes at once

**Mitigation**:
- Make changes incrementally
- Deprecate before deleting
- Maintain backward compatibility where possible
- Document migration paths

**Risk**: Breaking existing workflows

**Mitigation**:
- Keep old interfaces alongside new
- Add warnings for deprecated usage
- Provide migration guide
- Test all demos still work

---

## Next Steps

Axis 4 runs continuously throughout the improvement process:
- **Weeks 1-3**: Phase 1 (parallel with Axis 1)
- **Weeks 4-8**: Phase 2 (parallel with Axis 2)
- **Weeks 9-12**: Phase 3 (parallel with Axis 3)
- **Weeks 13-16**: Phase 4 (documentation)

After Week 16: Maintenance mode with continuous improvement

---

## Documents in This Axis

- **README.md** (this file) - Overview and action items
- **legacy-analysis.md** - Detailed legacy code analysis with patterns
- **modernization-plan.md** - Step-by-step modernization guide
- **dx-improvements.md** - Developer experience improvements
- **documentation-strategy.md** - Documentation improvement plan

## Related Documentation

- `../improvement-plan/README.md` - Overall coordination
- All other axes - This axis supports all improvements

---

**Questions?** See `modernization-plan.md` for detailed step-by-step guides and `dx-improvements.md` for developer experience improvements.
