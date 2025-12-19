# Recommended Extensions from Community Resources

> **Generated**: Dec 18, 2024
> **Sources**: Actual GitHub repositories and MCP server directories
> **Status**: Ready to install

This document contains **real, production-ready extensions** found from exploring:
- wshobson/agents (22K+ ⭐, 91 agents, 47 skills)
- rshah515/claude-code-subagents (133+ subagents)
- punkpeye/awesome-mcp-servers (74K+ ⭐)
- iRahulPandey/mlflowMCPServer (MLflow integration)

---

## Recommended Installation Order

### 🚀 High Priority (Install First)

1. **Python Development Plugin** (wshobson/agents)
2. **Test Automator Agent** (wshobson/agents)
3. **MLflow MCP Server** (for experiment tracking)

### ⚡ Medium Priority (Install After Setup)

4. **Data Scientist Agent** (wshobson/agents)
5. **Python-specific subagents** (rshah515 collection)

### 📚 Low Priority (Optional)

6. **Additional ML workflow agents**
7. **Custom skills for constitutional AI patterns**

---

## 1. Python Development Plugin (wshobson/agents)

### What It Provides

**Complete Python development ecosystem** with:
- 3 specialized agents (python-pro, django-pro, fastapi-pro)
- 5 skills (auto-activate when relevant)
- 1 scaffolding command

### Skills Included

| Skill | Description | Auto-Activates When |
|-------|-------------|---------------------|
| **async-python-patterns** | AsyncIO, concurrent programming, async/await for high-performance | Building async APIs, I/O-bound apps |
| **python-testing-patterns** | pytest, fixtures, mocking, TDD | Writing tests, test setup |
| **python-packaging** | setup.py/pyproject.toml, PyPI publishing | Creating packages, distribution |
| **python-performance-optimization** | cProfile, memory profilers, bottlenecks | Debugging slow code, optimization |
| **uv-package-manager** | Fast dependency management with uv | Setting up projects, managing deps |

### Installation

```bash
# Clone the repository
git clone https://github.com/wshobson/agents.git

# Copy python-development plugin to your project
cp -r agents/plugins/python-development /Users/apa/ml_projects/constitutional-ai/.claude/plugins/

# Or install globally
cp -r agents/plugins/python-development ~/.claude/plugins/
```

### Token Efficiency

- **Loaded**: ~300 tokens (3 agents + 5 skill metadata)
- **Activated**: Skills load full content only when triggered
- **Progressive disclosure**: Efficient even with 5 skills installed

### Why This Over Custom Skills

✅ **Production-tested** (22K+ stars, active maintenance)
✅ **Complete ecosystem** (not just one skill)
✅ **Modern tools** (uv, ruff, pydantic - 2024/2025)
✅ **Progressive disclosure** (efficient token usage)

---

## 2. Test Automator Agent (wshobson/agents)

### What It Provides

**AI-powered test automation specialist** for:
- Modern testing frameworks (pytest, Jest)
- Self-healing test automation
- Comprehensive quality engineering
- CI/CD integration

### Capabilities

```yaml
---
name: test-automator
description: Master AI-powered test automation with modern frameworks, self-healing tests, and comprehensive quality engineering. Build scalable testing strategies with advanced CI/CD integration. Use PROACTIVELY for testing automation or quality assurance.
tools: Read, Write, Edit, Glob, Grep, Bash
model: sonnet
---
```

### Key Features

1. **Pytest expertise** - Fixtures, parametrize, mocking, coverage
2. **Test generation** - Automatic test creation with edge cases
3. **Self-healing tests** - Adapts to code changes
4. **CI/CD integration** - GitHub Actions, GitLab CI
5. **Coverage analysis** - pytest-cov optimization

### Installation

```bash
# Copy test-automator agent
cp agents/test-automator.md /Users/apa/ml_projects/constitutional-ai/.claude/agents/
```

### When to Use

- Creating test suites
- Fixing failing tests
- Improving test coverage
- Setting up test automation in CI

### Complements Your Project

Your constitutional-ai project has:
- 658 tests across 14 modules
- pytest with coverage
- CI testing on 3 Python versions

Test-automator can help:
- Increase coverage from 45% → 60%+
- Generate edge case tests
- Optimize test performance
- Fix flaky tests

---

## 3. MLflow MCP Server (iRahulPandey)

### What It Provides

**Natural language interface to MLflow** for:
- Experiment tracking
- Model registry exploration
- Run comparison
- Metrics visualization

### Installation

#### Method 1: Automatic (Recommended)

```bash
npx -y @smithery/cli install @iRahulPandey/mlflowMCPServer --client claude
```

#### Method 2: Manual

```bash
# Clone repository
git clone https://github.com/iRahulPandey/mlflowMCPServer.git
cd mlflowMCPServer

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install mcp[cli] langchain-mcp-adapters langchain-openai langgraph mlflow
```

### Configuration

Add to Claude Code settings (`~/.claude/settings.json` or `~/.config/claude/settings.json`):

```json
{
  "mcpServers": {
    "mlflow": {
      "command": "python",
      "args": ["path/to/mlflowMCPServer/mlflow_server.py"],
      "env": {
        "MLFLOW_TRACKING_URI": "http://localhost:5000"
      }
    }
  }
}
```

### Usage Examples

Once configured, ask Claude:

```
"Show me the last 5 experiments"
"Compare runs from experiment 'constitutional-ai-training'"
"What are the best hyperparameters for experiment X?"
"Show metrics for run abc123"
```

### Integration with Constitutional AI

Perfect for tracking:
- Critique-revision training runs
- Reward model experiments
- PPO optimization iterations
- Principle evaluation metrics

### Setup Requirements

1. MLflow tracking server running
2. Python environment with MLflow
3. MCP server configured in Claude Code

---

## 4. Data Scientist Agent (wshobson/agents)

### What It Provides

**Expert data scientist** for:
- Advanced analytics
- Machine learning modeling
- Statistical analysis
- Business intelligence

### Capabilities

```yaml
---
name: data-scientist
description: Expert data scientist for advanced analytics, machine learning, and statistical modeling. Handles complex data analysis, predictive modeling, and business intelligence. Use PROACTIVELY for data analysis tasks, ML modeling, statistical analysis, and data-driven insights.
tools: Read, Write, Edit, Glob, Grep, Bash
model: sonnet
---
```

### Key Skills

1. **Data analysis** - Pandas, NumPy, statistical methods
2. **ML modeling** - scikit-learn, PyTorch, TensorFlow
3. **Visualization** - matplotlib, seaborn, plotly
4. **Feature engineering** - Data preprocessing, transformation
5. **Model evaluation** - Metrics, validation, interpretation

### Installation

```bash
cp agents/data-scientist.md /Users/apa/ml_projects/constitutional-ai/.claude/agents/
```

### Use Cases for Constitutional AI

- Analyzing principle violation patterns
- Statistical evaluation of critique effectiveness
- A/B testing different constitutional principles
- Analyzing reward model convergence
- Feature importance for harm detection

---

## 5. Python-Pro Agent (rshah515 collection)

### What It Provides

**Alternative Python expert** from rshah515's collection:
- 133+ specialized subagents
- Full software development lifecycle coverage
- Domain-specific expertise

### Python-Specific Agents

From the collection:
- `python-expert.md` - General Python development
- `testing-specialist.md` - Testing focus
- `ml-engineer.md` - Machine learning
- `data-engineer.md` - Data pipelines

### Installation

```bash
# Clone repository
git clone https://github.com/rshah515/claude-code-subagents.git

# Copy relevant agents
cp claude-code-subagents/language-specialists/python-expert.md /Users/apa/ml_projects/constitutional-ai/.claude/agents/
```

### Comparison: wshobson vs rshah515

| Feature | wshobson/agents | rshah515 subagents |
|---------|----------------|-------------------|
| **Total resources** | 91 agents, 47 skills | 133+ subagents |
| **Organization** | Plugins (bundled) | Individual agents |
| **Stars** | 22K+ | Smaller community |
| **Maintenance** | Very active | Active |
| **Focus** | Production-ready plugins | Comprehensive collection |

**Recommendation**: Start with wshobson for Python development (it's a complete plugin), use rshah515 for specialized agents not in wshobson.

---

## 6. Additional MCP Servers (punkpeye/awesome-mcp-servers)

### Python Runtime & Execution

| Server | Description | Use Case |
|--------|-------------|----------|
| **hileamlakB/PRIMS** | Python Runtime Interpreter | Execute Python in isolated env |
| **pydantic/pydantic-ai** | Run Python in sandbox | Safe code execution |
| **kestra-io/mcp-server** | Workflow orchestration | Training pipelines |

### Installation

Check individual repositories for setup instructions. Most follow MCP standard:

```json
{
  "mcpServers": {
    "server-name": {
      "command": "npx",
      "args": ["-y", "@org/server-name"]
    }
  }
}
```

---

## Implementation Roadmap

### Week 1: Core Setup ✅ (Already Done)

- [x] CLAUDE.md created
- [x] Black formatting hook
- [x] /test command
- [x] Basic skills created

### Week 2: Community Resources (Recommended)

- [ ] Install **python-development plugin** from wshobson
  - Replaces custom python-ml-patterns skill with production-tested version
  - Adds 4 more skills (async, packaging, performance, uv)

- [ ] Install **test-automator agent** from wshobson
  - Complements custom pytest-testing skill
  - Adds AI-powered test generation

- [ ] Set up **MLflow MCP server**
  - Start tracking experiments
  - Query runs via natural language

### Week 3: Specialization (Optional)

- [ ] Install **data-scientist agent** (if doing analysis)
- [ ] Add **python-expert** from rshah515 (if wshobson insufficient)
- [ ] Explore additional MCP servers (workflow orchestration, etc.)

---

## Quick Comparison: Custom vs Community

| Resource | Custom (Created) | Community (Real) | Recommendation |
|----------|------------------|------------------|----------------|
| **Python ML Patterns** | ✅ Basic guide | 🌟 python-development plugin (5 skills) | **Replace** with community |
| **Pytest Testing** | ✅ Good coverage | 🌟 test-automator + python-testing-patterns | **Keep both** - complementary |
| **Type Hints** | ✅ Project-specific | ⚠️ No direct community equivalent | **Keep custom** |
| **Test Command** | ✅ Works well | ⚠️ No direct replacement | **Keep custom** |
| **Black Hook** | ✅ Essential | ⚠️ No direct replacement | **Keep custom** |

### Summary

- **Replace**: python-ml-patterns → python-development plugin
- **Add**: test-automator agent (complements existing)
- **Add**: MLflow MCP (new capability)
- **Keep**: Type hints skill, /test command, black hook

---

## Installation Commands (Quick Reference)

### 1. Clone Repositories

```bash
cd ~/Downloads
git clone https://github.com/wshobson/agents.git
git clone https://github.com/rshah515/claude-code-subagents.git
git clone https://github.com/iRahulPandey/mlflowMCPServer.git
```

### 2. Install Python Development Plugin

```bash
# Project-specific
cp -r ~/Downloads/agents/plugins/python-development \
  /Users/apa/ml_projects/constitutional-ai/.claude/plugins/

# Or global (all projects)
cp -r ~/Downloads/agents/plugins/python-development ~/.claude/plugins/
```

### 3. Install Test Automator

```bash
cp ~/Downloads/agents/test-automator.md \
  /Users/apa/ml_projects/constitutional-ai/.claude/agents/
```

### 4. Install Data Scientist

```bash
cp ~/Downloads/agents/data-scientist.md \
  /Users/apa/ml_projects/constitutional-ai/.claude/agents/
```

### 5. Configure MLflow MCP

```bash
cd ~/Downloads/mlflowMCPServer
python -m venv venv
source venv/bin/activate
pip install mcp[cli] langchain-mcp-adapters langchain-openai langgraph mlflow
```

Then add to Claude Code settings (see section 3 above).

---

## Resources

- **wshobson/agents**: https://github.com/wshobson/agents
- **rshah515/subagents**: https://github.com/rshah515/claude-code-subagents
- **awesome-mcp-servers**: https://github.com/punkpeye/awesome-mcp-servers
- **MLflow MCP**: https://github.com/iRahulPandey/mlflowMCPServer
- **VoltAgent subagents**: https://github.com/VoltAgent/awesome-claude-code-subagents

---

## Next Steps

1. **Review this document** - Decide which resources to install
2. **Clone repositories** - Download to ~/Downloads
3. **Install Week 2 resources** - python-development, test-automator
4. **Test each installation** - Verify skills activate, agents work
5. **Configure MLflow** - If you want experiment tracking
6. **Remove redundant custom skills** - Replace with community versions

All resources listed here are:
- ✅ Production-ready
- ✅ Actively maintained
- ✅ Community-validated (high stars/usage)
- ✅ Compatible with Claude Code
- ✅ Ready to install
