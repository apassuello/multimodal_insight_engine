# Resource Discovery Reference

> **Purpose**: Find, evaluate, and select the right extensions for any Claude Code project.
> **When to use**: Setting up new projects, adding capabilities, solving new problems.
> **Deep-dive sources**: MCP-RAG-Integration-Guide, Extensions-Decision-Guide, Extensions-Inventory-Guide

---

## Discovery Sources

### Primary Search Locations

| Resource Type | Primary Source | Secondary Sources |
|---------------|----------------|-------------------|
| **Skills** | SkillsMP.com (26K+ indexed) | awesome-claude-code, travisvn/awesome-claude-skills |
| **MCP Servers** | punkpeye/awesome-mcp-servers (74K+ ⭐) | modelcontextprotocol/servers (72K+ ⭐) |
| **Agents** | wshobson/agents (22K+ ⭐, 99 agents) | rshah515/claude-code-subagents (165 agents) |
| **Commands** | awesome-claude-code | kiliczsh/claude-cmd (184+ commands) |
| **Plugins** | anthropics/claude-code/plugins (official) | jeremylongshore/claude-code-plugins-plus (243 plugins) |
| **Hooks** | awesome-claude-code | Community examples in Extensions-Inventory-Guide |

### Search Strategy by Need

```
"I need to automate a repetitive task"
    → Search: Commands first, then Hooks for enforcement

"I need domain expertise encoded"
    → Search: Skills on SkillsMP, then wshobson/agents

"I need external service integration"
    → Search: MCP servers in awesome-mcp-servers

"I need guaranteed enforcement (not probabilistic)"
    → Solution: Hooks (deterministic), not Skills (LLM-decided)

"I need complex multi-step orchestration"
    → Search: Agents/Subagents in VoltAgent or rshah515
```

---

## Universal Evaluation Framework

### Quality Criteria (applies to all resource types)

| Factor | Weight | Threshold | How to Assess |
|--------|--------|-----------|---------------|
| **Community validation** | Medium | 100+ stars | GitHub stars/forks |
| **Active maintenance** | High | < 3 months | Recent commits, issue responsiveness |
| **Documentation quality** | High | Clear README | Usage examples, configuration docs |
| **Production usage** | High | Referenced | Blog posts, discussions, real-world cases |
| **Compatibility** | Critical | Current version | Works with your Claude Code version |

### The "5-10 Rule" for Custom Builds

**Build custom only when**:
- You've done the task **at least 5 times** already
- You expect to do it **10+ more times**
- Below this threshold → maintenance overhead exceeds benefit

### Custom vs Existing Decision Tree

```
Does an existing solution exist?
├── NO → Build custom
└── YES → Quality check (stars, maintenance, docs)
    ├── LOW quality → Find better OR build custom
    └── HIGH quality → Does it cover 80%+ of need?
        ├── YES → Use existing
        └── NO (50-80%) → Evaluate: Fork cost vs custom build
```

---

## Skills

### What Skills Are
Model-invoked knowledge activated automatically based on context. Claude decides when to use them.

### When to Use Skills vs Other Types

| Situation | Use Skill? | Alternative |
|-----------|------------|-------------|
| Domain expertise needed automatically | ✅ Yes | - |
| Explicit user action required | ❌ No | Command |
| Guaranteed execution needed | ❌ No | Hook |
| External API access needed | ❌ No | MCP Server |

### Skill Evaluation Criteria

| Criterion | Good Sign | Red Flag |
|-----------|-----------|----------|
| **Description** | Specific trigger conditions, <200 chars | Vague ("helps with code") |
| **Activation rate** | Triggers reliably when relevant | Rarely activates or over-activates |
| **Size** | Concise, focused | >500 lines without good reason |
| **Failed attempts documented** | Yes, for complex domains | Missing for error-prone tasks |

### Key Skill Repositories

| Repository | Specialty | Notable Skills |
|------------|-----------|----------------|
| **wshobson/agents** | General development | 107 skills: Python, JS/TS, K8s, Cloud |
| **obra/superpowers** | Process enforcement | TDD, systematic-debugging, planning |
| **anthropics/skills** | Document handling | DOCX, PDF, PPTX, XLSX extraction |
| **daymade/claude-code-skills** | Production-ready | 23 skills + skill-creator meta-skill |

> **Deep dive**: Extensions-Decision-Guide, "Skills architecture: Five proven patterns"

---

## Commands

### What Commands Are
User-invoked shortcuts triggered by `/command`. Explicit actions, not automatic.

### When to Use Commands vs Other Types

| Situation | Use Command? | Alternative |
|-----------|--------------|-------------|
| Explicit user action, simple workflow | ✅ Yes | - |
| Complex multi-step procedures | ❌ No | Skill |
| Automatic context-based activation | ❌ No | Skill |
| Enforcement without user action | ❌ No | Hook |

### Command Evaluation Criteria

| Criterion | Good Sign | Red Flag |
|-----------|-----------|----------|
| **Simplicity** | Single focused purpose | Multi-file complex workflows |
| **Arguments** | Uses $ARGUMENTS effectively | Hardcoded values |
| **Description** | Clear in frontmatter | Missing or vague |
| **Team sharing** | In .claude/commands/ (git-tracked) | Personal only |

### Recommended Command Limit
**3-5 simple commands maximum per project**. More indicates over-engineering.

### Key Command Repositories

| Repository | Commands | Focus |
|------------|----------|-------|
| **kiliczsh/claude-cmd** | 184+ | Interactive CLI, MCP integration |
| **ursisterbtw/ccprompts** | 70+ | Full lifecycle (12 phases) |
| **hikarubw/claude-commands** | 5 core | Daily workflows: init, check, push, handover, plan |

> **Deep dive**: Extensions-Decision-Guide, "Commands vs skills vs MCPs vs hooks"

---

## Hooks

### What Hooks Are
Event-triggered automation at lifecycle points. **Deterministic** - guaranteed to execute.

### When to Use Hooks vs Other Types

| Situation | Use Hook? | Alternative |
|-----------|-----------|-------------|
| Guaranteed enforcement needed | ✅ Yes | - |
| Auto-format on file save | ✅ Yes | - |
| Block dangerous operations | ✅ Yes | - |
| Domain expertise application | ❌ No | Skill |
| User-initiated workflows | ❌ No | Command |

### Hook Lifecycle Events Reference

| Event | Trigger | Can Block? | Common Use |
|-------|---------|------------|------------|
| **PreToolUse** | Before tool execution | Yes (exit 2) | Block dangerous commands |
| **PostToolUse** | After tool completes | No | Formatting, linting |
| **Stop** | Claude finishes | Yes* | Auto-commit, validation |
| **SessionStart** | Session begins | No | Load context, set env |
| **SessionEnd** | Session terminates | No | Cleanup, logging |
| **Notification** | Claude needs input | No | Desktop alerts |

*Full list of 10 events in Config-Management-Reference

### Hook Selection Criteria

| Use Case | Recommended Hook | Exit Code |
|----------|------------------|-----------|
| Auto-format code | PostToolUse + matcher | 0 (continue) |
| Block .env modifications | PreToolUse + matcher | 2 (block) |
| Desktop notification | Stop or Notification | 0 |
| Load git context | SessionStart | 0 |

> **Deep dive**: Extensions-Inventory-Guide, "Claude Code hooks provide deterministic automation"

---

## MCP Servers

### What MCP Servers Are
External integrations connecting Claude to tools, databases, and APIs via Model Context Protocol.

### Platform Compatibility Matrix

| Server Type | Claude Code | Desktop | Web (Pro+) |
|-------------|:-----------:|:-------:|:----------:|
| Local stdio (filesystem, databases) | ✅ | ✅ | ❌ |
| Remote HTTP (GitHub, Atlassian) | ✅ | ✅ | ✅ |
| Desktop Extensions (.mcpb) | ❌ | ✅ | ❌ |

**Critical insight**: Local servers require process spawning → impossible in web browser sandbox.

### MCP Server Categories

| Category | Key Servers | Setup Complexity |
|----------|-------------|------------------|
| **Databases** | PostgreSQL, MySQL, MongoDB, Redis, SQLite | Simple-Medium |
| **Vector DBs** | Qdrant ✅, Pinecone, Chroma ✅, Milvus ✅, Weaviate | Medium |
| **Development** | GitHub ✅, Git, Filesystem | Simple |
| **Cloud** | AWS, Azure, Kubernetes, Terraform | Medium-Complex |
| **Communication** | Slack, Discord, Gmail | Medium |
| **Productivity** | Notion, Atlassian, Asana, Linear | Simple-Medium |

✅ = Official MCP server available

### MCP Evaluation Criteria

| Criterion | Good Sign | Red Flag |
|-----------|-----------|----------|
| **Source** | Official or verified publisher | Unknown author, no reviews |
| **Auth method** | Environment variables, OAuth | Hardcoded credentials |
| **Context cost** | Documented tool count | Unknown (could consume 14K+ tokens) |
| **Rate limits** | Documented | Undocumented |

### Context Budget Warning
A single MCP server with 20 tools can consume **14,000+ tokens**. One user reported **82,000 tokens (41% of context)** consumed by MCP tools before any conversation.

> **Deep dive**: MCP-RAG-Integration-Guide, "Part 1: MCP server catalog by category"

---

## Agents & Subagents

### What Agents Are
Specialized AI assistants with independent context windows, domain-specific prompts, and granular tool permissions.

### When Agents Are Appropriate

| ✅ Use Agents When | ❌ Agents Are Overkill When |
|-------------------|---------------------------|
| Multi-step tasks with planning cycles | Simple bug fixes |
| Large codebase refactors | Single-file edits |
| Test coverage campaigns | Quick lookups |
| Framework migrations | Well-defined narrow tasks |
| Security assessments | Single tool call sufficient |

### Built-in Subagent Types

| Subagent | Model | Purpose | Tools |
|----------|-------|---------|-------|
| **General** | Sonnet | Complex multi-step tasks | All tools |
| **Plan** | Sonnet | Research in plan mode | Read, Glob, Grep, Bash |
| **Explore** | Haiku | Fast read-only search | Glob, Grep, Read |

### Key Agent Repositories

| Repository | Agents | Specialty |
|------------|--------|-----------|
| **wshobson/agents** | 99 | Full SDLC coverage |
| **rshah515/claude-code-subagents** | 165 | Broadest domain coverage (including embedded, quantum) |
| **VoltAgent/awesome-claude-code-subagents** | 100+ | Production-ready with granular permissions |
| **avivl/claude-007-agents** | 112 | 14 categories with Task Master integration |

> **Deep dive**: Extensions-Decision-Guide, "Agent patterns: When autonomous makes sense"

---

## Plugins

### What Plugins Are
Bundled collections of skills, commands, hooks, and/or agents distributed as a package.

### Official Anthropic Plugins

| Plugin | Features | Quality |
|--------|----------|---------|
| **code-review** | Multi-agent PR review (5 parallel Sonnet agents) | Production |
| **feature-dev** | /feature-dev command with architect, reviewer agents | Production |
| **security-auditor** | PreToolUse hook monitoring 9 security patterns | Production |
| **frontend-design** | Auto-invoked skill for frontend work | Stable |
| **plugin-dev** | 7 specialized skills for plugin development | Production |

### Plugin Evaluation

| Criterion | Good Sign | Red Flag |
|-----------|-----------|----------|
| **Source** | Official or high-star community | Unknown publisher |
| **Composition** | Clear what's included (skills, hooks, etc.) | Opaque bundle |
| **Overlap** | Unique capabilities | Duplicates existing tools |
| **Documentation** | Install + usage + configuration | Missing |

> **Deep dive**: Extensions-Inventory-Guide, "Official Anthropic plugins form the foundation"

---

## Reference Files

### What Reference Files Are
Documentation stored in .claude/docs/ for on-demand reading. Not automatically loaded.

### When to Use Reference Files

| Use Case | Reference File? | Alternative |
|----------|-----------------|-------------|
| Large catalogs/inventories | ✅ Yes | - |
| Detailed documentation | ✅ Yes | - |
| Frequently needed info | ❌ No | CLAUDE.md or Skill |
| Every-prompt context | ❌ No | CLAUDE.md |

### Organization Pattern

```
.claude/
├── docs/
│   ├── resource-discovery-reference.md   ← This document
│   ├── config-management-reference.md    ← Companion document
│   └── [domain-specific-docs]/
```

---

## Quick Selection Matrix

| I need... | Primary Type | Secondary |
|-----------|--------------|-----------|
| Domain expertise (automatic) | Skill | - |
| Explicit user action | Command | - |
| Guaranteed enforcement | Hook | - |
| External API/service access | MCP Server | - |
| Complex multi-step orchestration | Agent | Skill |
| Bundled capabilities | Plugin | - |
| Large reference material | Reference File | - |

---

## Version History

- v1.0.0 (2024-12): Initial release based on ecosystem analysis
