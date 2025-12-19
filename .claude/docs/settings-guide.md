# Project Settings Guide

> **Purpose**: Explains the team-shared `.claude/settings.json` configuration
> **Created**: 2025-12-19
> **Applies to**: All team members working on MultiModal Insight Engine

---

## Overview

The `.claude/settings.json` file configures permissions, environment variables, and workflows for this ML project. It's version-controlled and shared across the team.

**Location**: `.claude/settings.json` (git-tracked)
**Personal overrides**: `.claude/settings.local.json` (git-ignored)

---

## Permissions Configuration

### ✅ Allowed Operations

**Testing & Development:**
```json
"Bash(python -m pytest*)"     // Run all pytest commands
"Bash(python -m coverage*)"   // Generate coverage reports
"Bash(python src/*)"          // Run project Python scripts
```

**Git Operations:**
```json
"Bash(git status)"            // Check repository status
"Bash(git diff*)"             // View changes
"Bash(git log*)"              // View commit history
"Bash(git add*)"              // Stage files
"Bash(git commit*)"           // Create commits
```

**File Operations:**
```json
"Bash(ls*)"                   // List files
"Bash(cat*)"                  // Read files
"Bash(grep*)"                 // Search in files
"Bash(find*)"                 // Find files
"Bash(wc*)"                   // Count lines/words
"Bash(head*)"                 // View file beginning
"Bash(tail*)"                 // View file end
```

### ❌ Denied Operations

**Dangerous Commands:**
```json
"Bash(rm -rf *)"              // Prevent accidental deletion
"Bash(sudo *)"                // Block privileged operations
"Bash(pip install *)"         // Require explicit approval for dependencies
"Bash(pip uninstall *)"       // Require explicit approval for removals
```

**Why deny pip install/uninstall?**
- Prevents unexpected dependency changes
- Ensures team discusses new dependencies
- Maintains reproducible environment
- Claude will ask permission when needed

---

## Environment Variables

### PYTHONPATH
```json
"PYTHONPATH": "src"
```
- Adds `src/` to Python module search path
- Allows importing from `src/` without installation
- Example: `from models.transformer import MultiModalModel`

### PYTORCH_ENABLE_MPS_FALLBACK
```json
"PYTORCH_ENABLE_MPS_FALLBACK": "1"
```
- Enables Metal Performance Shaders fallback on macOS
- Prevents crashes when MPS operations aren't supported
- Falls back to CPU for unsupported operations

---

## Hooks

Currently no hooks configured. Hooks are for automation like:
- Auto-formatting code on save
- Running linters before commits
- Validating test coverage

**To add hooks**, see `.claude/docs/config-management-reference.md`

---

## MCP Servers

### GitHub MCP Server ✓

**Status**: Installed and connected (local scope)

**Capabilities:**
- Create/manage GitHub issues and PRs
- Search repositories and code
- View commits, branches, and repository info
- Automate GitHub workflows

**Documentation**: See `.claude/docs/github-mcp-guide.md` for complete usage guide

**Configuration**: Stored in `~/.claude.json` under this project path

### Other MCP Servers

Additional MCP servers can be added for:
- Database connections (PostgreSQL, MongoDB, etc.)
- Cloud service access (AWS, Azure, GCP)
- API integrations (Slack, Notion, etc.)

**User-level MCP servers** (in `~/.claude.json`) still work across all projects.

---

## Customizing for Personal Workflow

Create `.claude/settings.local.json` for personal overrides (never commit this!):

```json
{
  "permissions": {
    "allow": [
      "Bash(python -m ipython)"
    ]
  },
  "env": {
    "CUSTOM_VAR": "value"
  }
}
```

**Precedence**: `.claude/settings.local.json` > `.claude/settings.json` > `~/.claude/settings.json`

---

## Common Scenarios

### 1. "I need to install a new Python package"

Claude will ask for permission when running `pip install`. Approve it, then:
1. Update `requirements.txt` or `pyproject.toml`
2. Document why the dependency was added
3. Commit the change

### 2. "Claude is asking permission for every git command"

Your `.claude/settings.local.json` might be overriding. Check:
```bash
cat .claude/settings.local.json
```

### 3. "I want to add auto-formatting"

Add a PostToolUse hook in this file:
```json
"hooks": {
  "PostToolUse": [
    {
      "matcher": "Edit|Write",
      "hooks": [
        {
          "type": "command",
          "command": "black \"$file\"",
          "timeout": 30
        }
      ]
    }
  ]
}
```

### 4. "I want different settings for my machine"

Use `.claude/settings.local.json` - it's already gitignored!

---

## Security Best Practices

✅ **DO:**
- Keep this file in version control
- Document permission changes
- Use environment variables for paths
- Deny dangerous operations

❌ **DON'T:**
- Add API keys or secrets here
- Allow unrestricted sudo/rm commands
- Hardcode personal file paths
- Override team security policies

**For secrets**: Use environment variables or `.env` files (gitignored)

---

## Maintenance

**When to update this file:**
- Adding new common workflows
- Team agrees on new tool permissions
- Environment variable requirements change
- Security policies evolve

**Review schedule:**
- Quarterly review of permissions
- After onboarding new team members
- When adding major new features

---

## Troubleshooting

### "Permission denied" errors

**Symptom**: Claude asks permission repeatedly for the same command

**Solutions:**
1. Check if command matches allow patterns
2. Add pattern to `.claude/settings.local.json` (personal)
3. Propose adding to team settings (this file)

### Settings not loading

**Solutions:**
1. Verify JSON syntax: `python -m json.tool .claude/settings.json`
2. Restart Claude Code
3. Check for syntax errors in settings.local.json

### Conflicting permissions

**Precedence order** (highest to lowest):
1. `.claude/settings.local.json` (personal)
2. `.claude/settings.json` (this file - team)
3. `~/.claude/settings.json` (global user settings)

**Note**: Deny rules ALWAYS win over allow rules at any level!

---

## Version History

- v1.0.0 (2025-12-19): Initial settings for ML project
  - Basic permissions for testing, git, file operations
  - PYTHONPATH and MPS fallback configured
  - Security denials for dangerous operations
