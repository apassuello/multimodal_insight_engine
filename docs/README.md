# Documentation Index

This directory contains all project documentation organized by category.

## Directory Structure

### **improvement-plan/** - Improvement Roadmap
Complete 10-16 week improvement plan with 4 axes:
- [`1-security-and-stability/`](improvement-plan/1-security-and-stability/) - Critical security fixes (Weeks 1-2)
- [`2-architecture-refactoring/`](improvement-plan/2-architecture-refactoring/) - Architecture improvements (Weeks 3-6)
- [`3-testing-and-quality/`](improvement-plan/3-testing-and-quality/) - Testing & CI/CD (Weeks 7-10)
- [`4-repository-structure/`](improvement-plan/4-repository-structure/) - Modernization (Ongoing)
- [`diagrams/`](improvement-plan/diagrams/) - Visual architecture diagrams

**Start here**: [`improvement-plan/README.md`](improvement-plan/README.md)

### **demo/** - Demo Documentation
Constitutional AI demo documentation:
- `DEMO_ARCHITECTURE.md` - Interactive demo architecture
- `DEMO_AUDIT_REPORT.md` - Demo audit findings
- `DEMO_IMPROVEMENT_PLAN.md` - Original improvement plan
- `DEMO_IMPROVEMENT_PLAN_V2.md` - V2 with model research
- `DEMO_IMPROVEMENT_PLAN_V3.md` - V3 with realistic model sizes
- `DEMO_PERFORMANCE_ANALYSIS.md` - Performance analysis
- `DUAL_MODEL_SETUP.md` - Dual model configuration guide
- `UI_LAYOUT.md` - UI layout documentation

### **deployment/** - Deployment Documentation
Production deployment guides:
- `DEPLOYMENT.md` - Deployment guide
- `DEPLOYMENT_FIXES_SUMMARY.md` - Deployment fixes
- `DOCKER_DEPLOYMENT_PLAN.md` - Docker deployment plan

### **audits/** - Code Reviews & Audits
Code review and audit reports:
- `PYTHON_CODE_REVIEW.md` - Python code review
- `VERIFICATION_REPORT.md` - Verification results
- `AGENT_VALIDATION_REPORT.md` - Agent validation

### **security/** - Security Audits
Security audit documentation:
- `SECURITY_AUDIT_PHASE2.md` - Phase 2 security audit
- `SECURITY_FIXES_PHASE2.md` - Phase 2 security fixes
- `SECURITY_FIXES_SUMMARY.md` - Security fixes summary

### **assessments/** - Repository Assessments
Historical assessment reports:
- `AUDIT_FINDINGS.md` - Independent code audit results
- `code_quality_assessment.md` - Code quality metrics
- `MERGE_READINESS_ASSESSMENT.md` - Merge status checklist
- `current_test_status.md` - Test coverage status
- `test_implementation_plan.md` - Testing plan

### **reference/** - Reference Documentation
Technical reference materials and educational content:

**Core Architecture:**
- `models_architecture.md` - Model architecture documentation
- `optimization_architecture.md` - Optimization algorithms
- `training_architecture.md` - Training pipeline architecture
- `data_architecture.md` - Data loading and processing
- `project_architecture.md` - Overall architecture overview

**Concepts & Theory:**
- `language_model_concepts.md` - Language modeling fundamentals
- `image_processing_concepts.md` - Image processing and vision
- `attention_mechanisms.md` - Attention mechanisms explained
- `neural_network_fundamentals.md` - Neural network basics
- `neural_network_foundations.md` - Foundation layer documentation
- `qkv_projections_guide.md` - Query-Key-Value projections

**Educational Content:**
- `anthropic_insights.md` - Anthropic research and Constitutional AI
- `learning_techniques.md` - Advanced learning techniques

**Practical Guides:**
- `training_insights.md` - Training tips and best practices
- `transformer_training_debug.md` - Debugging transformer training
- `testing_documentation.md` - Testing infrastructure guide
- `hardware_profiling.md` - Performance profiling tools

**Demos:**
- `demos_overview.md` - Demo scripts index
- `language_model_demo.md` - Language model demo
- `DEMO_GUIDE.md` - Constitutional AI demo guide

**Other:**
- `README_tokenization.md` - Tokenization documentation
- `code_directory.md` - Code implementations index
- `claude-context.md` - Claude Code integration context
- `metadata_prompt.md` - Metadata generation guide

### **constitutional-ai/** - Constitutional AI Documentation
Complete Constitutional AI implementation documentation:
- `CONSTITUTIONAL_AI_ARCHITECTURE.md` - System architecture
- `CONSTITUTIONAL_AI_IMPLEMENTATION_SPEC.md` - Implementation specification
- `PPO_IMPLEMENTATION_GUIDE.md` - PPO training guide
- `REWARD_MODEL_IMPLEMENTATION_SUMMARY.md` - Reward model implementation
- `PROMPT_GENERATION_GUIDE.md` - Prompt generation strategies
- `CONSTITUTIONAL_AI_TEST_COVERAGE.md` - Test coverage details
- `PPO_VERIFICATION.md` - PPO verification results
- `PPO_IMPLEMENTATION_SUMMARY.md` - PPO implementation summary

### **testing/** - Testing Documentation
Testing infrastructure and practices:
- `TESTING_QUICK_REFERENCE.md` - Quick reference for testing

### **archive/** - Archived Documents
Historical or superseded documentation:

**Phase Archives:**
- `phase2/` - Phase 2 implementation summaries
- `phase3/` - Phase 3 completion and verification reports
- `constitutional-ai-dev/` - Constitutional AI development history
- `todos/` - Old TODO lists

**Legacy Documentation (from doc/ migration):**
- `legacy/sds/` - Software Design Specifications (9 files)
- `legacy/demos/` - Old demo documentation (5 files)
- `legacy/misc/` - Miscellaneous legacy docs
- `legacy/code-samples/` - Python reference implementations (27 files)

**Other:**
- `Multimodal Training Challenge.md` - Training challenges documentation

---

## Quick Navigation

**New to the project?**
1. Root: [`README.md`](../README.md) - Project overview
2. Root: [`GETTING_STARTED.md`](../GETTING_STARTED.md) - Setup guide
3. [`ARCHITECTURE.md`](ARCHITECTURE.md) - System architecture
4. Root: [`CLAUDE.md`](../CLAUDE.md) - Development guidelines

**Developer resources:**
1. [`API_REFERENCE.md`](API_REFERENCE.md) - Complete API documentation
2. [`USER_GUIDE.md`](USER_GUIDE.md) - Constitutional AI Demo guide
3. Root: [`CONTRIBUTING.md`](../CONTRIBUTING.md) - How to contribute
4. Root: [`CHANGELOG.md`](../CHANGELOG.md) - Version history

**Want to improve the codebase?**
1. [`improvement-plan/README.md`](improvement-plan/README.md) - Improvement plan overview
2. Axes 1-4 in [`improvement-plan/`](improvement-plan/) - Detailed roadmaps

**Security & Conduct:**
1. Root: [`SECURITY.md`](../SECURITY.md) - Security policy
2. Root: [`CODE_OF_CONDUCT.md`](../CODE_OF_CONDUCT.md) - Community guidelines

**Looking for specific docs?**
- Demo documentation → `demo/`
- Deployment guides → `deployment/`
- Code audits → `audits/`
- Security audits → `security/`
- Improvement plans → `improvement-plan/`
- Past assessments → `assessments/`
- Technical reference → `reference/`
- Constitutional AI → `constitutional-ai/`
- Testing docs → `testing/`
- Old documents → `archive/`

---

## Documentation Organization Principles

1. **Root stays clean** - Only 7 essential files (README, GETTING_STARTED, CLAUDE, CONTRIBUTING, CHANGELOG, SECURITY, CODE_OF_CONDUCT)
2. **Logical grouping** - Documents organized by purpose (demo, deployment, audits, security, etc.)
3. **Clear navigation** - README files guide the way
4. **Archival strategy** - Old docs moved to `archive/`
5. **Self-contained** - Each section has what it needs

---

**Questions?** See [`improvement-plan/README.md`](improvement-plan/README.md) for the detailed roadmap.
