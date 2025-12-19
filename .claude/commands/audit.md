---
description: Audit Claude Code configuration for issues, optimization opportunities, and best practice compliance
allowed-tools: Read, Bash(cat:*), Bash(ls:*), Bash(find:*), Glob
argument-hint: [scope: all | skills | commands | hooks | mcp | performance | security]
---

# Configuration Audit

Read `.claude/docs/config-management-reference.md` for the audit checklist and best practices.

## Audit Scope
$ARGUMENTS

If no scope specified, perform full audit.

## Audit Procedure

### 1. Gather Current State

**Files to check:**
- CLAUDE.md (root)
- .claude/settings.json
- .claude/settings.local.json (if exists)
- .mcp.json (if exists)
- .claude/skills/*/SKILL.md
- .claude/commands/*.md

**Commands to run:**
- `ls -la .claude/` - Directory structure
- `cat CLAUDE.md | wc -l` - CLAUDE.md size
- List skill and command counts

### 2. Apply Checklist by Scope

**If scope includes "all" or "skills":**
- [ ] Skills have specific descriptions with trigger conditions
- [ ] No skills >500 lines without justification
- [ ] No vague descriptions ("helps with code")

**If scope includes "all" or "commands":**
- [ ] Command count ≤5
- [ ] Commands use $ARGUMENTS (no hardcoded values)
- [ ] All commands have description frontmatter

**If scope includes "all" or "hooks":**
- [ ] Hooks use script files for complex logic
- [ ] Exit codes are explicit
- [ ] No style enforcement hooks (use linters)

**If scope includes "all" or "mcp":**
- [ ] Server count <10
- [ ] No hardcoded credentials
- [ ] Servers from trusted sources

**If scope includes "all" or "performance":**
- [ ] CLAUDE.md <500 lines
- [ ] No code style rules in CLAUDE.md
- [ ] MCP context not excessive

**If scope includes "all" or "security":**
- [ ] Sensitive files in deny list (.env*, secrets/)
- [ ] Dangerous commands in deny list (rm -rf, sudo)
- [ ] API keys in environment variables only

### 3. Generate Report

## Output Format

```
# Configuration Audit Report

## Summary
- Overall health: [Good | Needs attention | Critical issues]
- Issues found: [count]
- Optimization opportunities: [count]

## Issues (Fix Required)

### [Issue 1]
- Location: [file/setting]
- Problem: [what's wrong]
- Fix: [specific action]

## Optimizations (Recommended)

### [Optimization 1]
- Current: [current state]
- Recommended: [better state]
- Impact: [what improves]

## Passed Checks
- [List of items that passed]

## Next Steps
1. [Prioritized action 1]
2. [Prioritized action 2]
```