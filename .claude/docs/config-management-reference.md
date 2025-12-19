# Configuration Management Reference

> **Purpose**: Configure, maintain, optimize, and troubleshoot Claude Code extensions.
> **When to use**: Setup, ongoing maintenance, performance tuning, troubleshooting.
> **Deep-dive sources**: Extensions-Operations-Playbook, Extensions-Inventory-Guide, MCP-RAG-Integration-Guide

---

## Configuration Architecture

### File Locations & Precedence

Settings resolve through strict hierarchy—**more specific overrides broader**:

| Precedence | Location | Scope | Git-tracked |
|------------|----------|-------|-------------|
| 1 (Highest) | `managed-settings.json` | Enterprise policies | No (IT-deployed) |
| 2 | Command-line arguments | Session-specific | N/A |
| 3 | `.claude/settings.local.json` | Personal project overrides | No |
| 4 | `.claude/settings.json` | Team-shared project settings | Yes |
| 5 (Lowest) | `~/.claude/settings.json` | Personal global defaults | No |

**Critical rule**: Deny rules always win over allow rules at any level.

### What Goes Where

| Content Type | Location | Rationale |
|--------------|----------|-----------|
| Project overview, critical constraints | CLAUDE.md | Needed every prompt |
| Team permissions, hooks | .claude/settings.json | Shared via git |
| Personal overrides | .claude/settings.local.json | Not committed |
| MCP servers (team) | .mcp.json | Shared via git |
| MCP servers (personal) | ~/.claude.json | All your projects |
| Skills (project) | .claude/skills/ | Project-specific expertise |
| Skills (personal) | ~/.claude/skills/ | Cross-project expertise |
| Commands (project) | .claude/commands/ | Team workflows |
| Commands (personal) | ~/.claude/commands/ | Personal shortcuts |
| Reference docs | .claude/docs/ | On-demand reading |

### MCP Configuration Scopes

| Scope | Location | Use Case |
|-------|----------|----------|
| Local (default) | ~/.claude.json under project path | Private, current project |
| Project | .mcp.json in project root | Shared via version control |
| User | ~/.claude.json mcpServers field | Available across all projects |

---

## Skills Management

### Lifecycle: Create → Test → Iterate → Maintain

**1. Create**
```
.claude/skills/[skill-name]/
└── SKILL.md    ← Required file
```

Frontmatter requirements:
```yaml
---
name: skill-name              # Identifier
description: Clear trigger conditions, max 200 chars
---
```

**2. Test Activation Separately from Execution**
- Activation test: Does it trigger when expected?
  - Not activating → broaden description, add use case keywords
  - Over-activating → narrow description, add constraints
- Execution test: Does it produce correct results?
  - Inconsistent → add specificity, include validation steps

**3. Iterate**
- Community finding: **803 lines → 400 lines with zero functionality lost**
- More lines ≠ better instructions
- Cut redundancy; add examples where behavior unclear

**4. Maintain**
- Add version history in SKILL.md
- Update when project patterns change
- Remove skills that no longer activate reliably

### Best Practices

| Practice | Implementation |
|----------|----------------|
| Specific descriptions | Include trigger conditions: "Use when [context]" |
| Progressive disclosure | Link to reference files for details |
| Failed attempts tables | Document what doesn't work for complex domains |
| Test across models | Haiku needs more guidance; Opus needs less |

### Anti-Patterns

| Anti-Pattern | Problem | Fix |
|--------------|---------|-----|
| Vague description | Claude can't determine relevance | Add specific trigger conditions |
| Token bloat | Loads unnecessarily, wastes context | Use progressive disclosure |
| Missing SKILL.md | Won't be recognized | Create required file |
| Over-engineering | Maintenance burden | Start minimal, expand iteratively |

---

## Commands Management

### Lifecycle

**1. Create**
```markdown
# .claude/commands/[command-name].md
---
description: What this command does
allowed-tools: Tool1, Tool2
argument-hint: [optional hint]
---

Command instructions here.
Use $ARGUMENTS for user input.
```

**2. Test**
- Invoke with `/command-name [args]`
- Verify correct interpretation of $ARGUMENTS
- Check tool permissions work as expected

**3. Maintain**
- Keep in .claude/commands/ for git tracking
- Document in team onboarding
- Remove unused commands

### Best Practices

| Practice | Implementation |
|----------|----------------|
| Simple and focused | One command, one purpose |
| Use $ARGUMENTS | Don't hardcode values |
| Clear descriptions | Frontmatter description required |
| Git-tracked | .claude/commands/ not personal-only |
| Limited count | 3-5 per project maximum |

### Anti-Patterns

| Anti-Pattern | Problem | Fix |
|--------------|---------|-----|
| Too complex | Multi-file workflows are skills | Move to skill |
| Hardcoded paths | Breaks on other machines | Use relative paths or env vars |
| Missing descriptions | SlashCommand tool won't discover | Add frontmatter |
| Too many commands | Over-engineering | Consolidate or use skills |

---

## Hooks Management

### All 10 Lifecycle Events

| Event | Trigger | Can Block? | Exit 2 Effect |
|-------|---------|------------|---------------|
| **UserPromptSubmit** | User submits prompt | Yes | Blocks prompt |
| **PreToolUse** | Before tool execution | Yes | Blocks tool |
| **PermissionRequest** | Permission dialog shown | Yes | Denies permission |
| **PostToolUse** | After tool completes | No | N/A |
| **Notification** | Claude needs input | No | N/A |
| **Stop** | Claude finishes responding | Yes* | Can continue |
| **SubagentStop** | Subagent completes | Yes* | Can continue |
| **PreCompact** | Before context compaction | No | N/A |
| **SessionStart** | Session begins/resumes | No | N/A |
| **SessionEnd** | Session terminates | No | N/A |

### Configuration Syntax

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "npx prettier --write \"$file\"",
            "timeout": 60
          }
        ]
      }
    ]
  }
}
```

### Matcher Patterns

| Pattern | Matches |
|---------|---------|
| `"Write"` | Exact tool name |
| `"Edit\|Write\|MultiEdit"` | Any of listed |
| `"*"` or `""` | All tools |
| `"mcp__github__.*"` | MCP server tools |

### Exit Codes

| Code | Meaning | Use For |
|------|---------|---------|
| 0 | Success, continue | PostToolUse, SessionStart |
| 2 | Block/deny | PreToolUse, PermissionRequest |
| Other | Non-blocking error | Stderr shown to user |

### Best Practices

| Practice | Implementation |
|----------|----------------|
| Script files for complex logic | Don't inline complex commands |
| Explicit exit codes | Return 2 to block, 0 to allow |
| Timeout configuration | Set reasonable limits |
| Test matchers carefully | Verify they match intended tools |

### Anti-Patterns

| Anti-Pattern | Problem | Fix |
|--------------|---------|-----|
| Complex inline commands | Hard to debug, maintain | Use script files |
| Missing exit codes | Unpredictable behavior | Always explicit exit |
| Untrusted code | Runs with your credentials | Review before enabling |
| Style enforcement hooks | Expensive, use linters | Move to deterministic tools |

---

## MCP Servers Management

### Installation by Transport

**HTTP/Remote servers:**
```bash
claude mcp add --transport http notion https://mcp.notion.com/mcp
```

**Local npm-based servers (stdio):**
```bash
claude mcp add --transport stdio github \
  --env GITHUB_PERSONAL_ACCESS_TOKEN=$GITHUB_TOKEN \
  -- npx -y @modelcontextprotocol/server-github
```

**Python-based servers:**
```bash
claude mcp add --transport stdio time \
  -- uvx mcp-server-time --local-timezone America/New_York
```

**Import from Claude Desktop:**
```bash
claude mcp add-from-claude-desktop
```

### Verification Commands

```bash
claude mcp list              # List all servers
claude mcp get github        # Get specific server config

# Within Claude Code sessions:
/mcp                         # View status, authenticate OAuth
/doctor                      # Comprehensive diagnostics
/context                     # Visual context breakdown
```

### Authentication Patterns

| Method | Use For | Security |
|--------|---------|----------|
| Environment variables | Most servers | ✅ Recommended |
| OAuth 2.0 | Remote services (GitHub, Atlassian) | ✅ Recommended |
| API tokens | Direct API access | ⚠️ Rotate regularly |
| Hardcoded | Never | ❌ Forbidden |

### Context Budget Management

**The Critical Constraint**: MCP tools consume context before you start working.

| Situation | Context Cost | Action |
|-----------|--------------|--------|
| Server with 20 tools | 14,000+ tokens | Consider selective enabling |
| Multiple servers | Can reach 82,000 tokens (41%) | Use McPick for toggling |
| `/doctor` warning | "Large MCP tools context" >25K | Optimize immediately |

**Optimization strategies:**
1. **Selective enabling**: Use `npx mcpick` to toggle servers per task
2. **Tool consolidation**: One tool with parameters vs. multiple similar tools
3. **Description optimization**: 87 tokens → 12 tokens with concise descriptions
4. **Server limit**: Keep under 10 active servers

### Troubleshooting

| Error | Cause | Solution |
|-------|-------|----------|
| `ECONNREFUSED` | Server not running | Verify server process |
| `spawn ENOENT` | Path not found | Use absolute paths |
| `Unexpected token` | Invalid JSON on stdout | Ensure logs → stderr only |
| `-32000: Connection closed` | Timeout/firewall | Increase MCP_TIMEOUT |
| Protocol version mismatch | Incompatible versions | Update MCP server/client |

**Debug command:**
```bash
claude --mcp-debug
```

### Best Practices

| Practice | Implementation |
|----------|----------------|
| Environment variables | Never hardcode credentials |
| Minimal permissions | Read-only when possible |
| Timeout configuration | Set MCP_TIMEOUT in settings |
| Regular verification | Run `/mcp` to check status |

### Anti-Patterns

| Anti-Pattern | Problem | Fix |
|--------------|---------|-----|
| Hardcoded credentials | Security risk | Use environment variables |
| Too many servers | Context exhaustion | Limit to <10 active |
| No timeout config | Hangs on failures | Set MCP_TIMEOUT |
| Wildcards in permissions | Not supported | Use explicit tool names |

---

## Agents Management

### Orchestration Patterns

**Sequential** (most common):
```
User → architect → developer → tester → reviewer → Result
```

**Parallel** (independent tasks):
```
User → [performance-eng + db-optimizer] → Merged result
```

**Conditional routing:**
```
User → analyzer → Routes to specialist based on analysis
```

### Tool Permissions

Define explicitly per agent:
```yaml
tools: Read, Grep, Glob, Bash
```

Avoid overly broad permissions for specialized agents.

### Best Practices

| Practice | Implementation |
|----------|----------------|
| Explicit tool permissions | List allowed tools per agent |
| Clear handoff points | Define when agent completes |
| Result validation | Verify agent output before accepting |
| Model selection | Haiku for fast tasks, Sonnet for complex |

---

## Performance Optimization

### Context Budget Strategies

| Strategy | Impact | Implementation |
|----------|--------|----------------|
| Selective MCP enabling | 60%+ savings | Use McPick |
| Tool consolidation | 60% reduction | Merge similar tools |
| Description optimization | 80%+ reduction | Concise descriptions |
| CLAUDE.md trimming | Variable | Remove verbose paragraphs |

### Startup Time Optimization

| Issue | Cause | Fix |
|-------|-------|-----|
| 2+ minute startup | Large ~/.claude.json | Clean old project metadata |
| Slow shell integration | Expensive .zshrc | Lazy load (nvm, etc.) |

**Lazy loading pattern (95% improvement):**
```bash
# ~/.zshrc
npm() {
    nvm use default --silent
    unfunction node npm npx
    npm "$@"
}
```

### Timeout Configuration

```json
{
  "env": {
    "MCP_TIMEOUT": "60000",
    "MCP_TOOL_TIMEOUT": "120000",
    "MAX_MCP_OUTPUT_TOKENS": "25000",
    "BASH_DEFAULT_TIMEOUT_MS": "1800000"
  }
}
```

---

## Audit & Maintenance

### Weekly Tasks

- [ ] Review context usage patterns (`/context`)
- [ ] Clear session data if accumulating
- [ ] Check for Claude Code updates

### Monthly Tasks

- [ ] Audit CLAUDE.md effectiveness; refine based on friction
- [ ] Review and update tool permissions
- [ ] Check extension updates; test before applying
- [ ] Clean ~/.claude.json of old project metadata
- [ ] Verify hooks still relevant

### Quarterly Tasks

- [ ] Full configuration audit using checklist below
- [ ] Remove unused extensions
- [ ] Review hook performance and necessity
- [ ] Update documentation for team onboarding

### Configuration Audit Checklist

**Essential Files:**
- [ ] CLAUDE.md exists at repository root
- [ ] CLAUDE.md is concise (<500 lines, no verbose paragraphs)
- [ ] No code style rules in CLAUDE.md (use linters instead)
- [ ] .claude/settings.json configured with appropriate permissions
- [ ] .claude/commands/ contains 3-5 simple shortcuts only
- [ ] .mcp.json configured if using external integrations

**Security:**
- [ ] API keys in environment variables, never hardcoded
- [ ] Sensitive files in deny list: `.env*`, `secrets/`, `credentials/`
- [ ] Dangerous commands in deny list: `rm -rf *`, `sudo *`, `curl * | sh`
- [ ] MCP servers from trusted sources only
- [ ] Permission review completed this quarter

**Performance:**
- [ ] ~/.claude.json size reasonable (startup <30s)
- [ ] Context management strategy documented
- [ ] MCP server count <10 active
- [ ] No complex hook chains adding overhead

**Team:**
- [ ] CLAUDE.md committed to version control
- [ ] .claude/commands/ shared with team
- [ ] .claude/settings.json shared (project-level permissions)
- [ ] Onboarding documentation exists

---

## Troubleshooting Decision Tree

```
Extension not working?
│
├── Skill not activating?
│   ├── Check description specificity
│   ├── Verify SKILL.md exists in correct location
│   └── Test with explicit context mention
│
├── Command not found?
│   ├── Check file location (.claude/commands/)
│   ├── Verify .md extension
│   └── Check frontmatter syntax
│
├── Hook not executing?
│   ├── Verify JSON syntax in settings.json
│   ├── Check matcher pattern matches tool
│   ├── Verify script is executable (chmod +x)
│   └── Check exit codes
│
├── MCP server not connecting?
│   ├── Listed in `claude mcp list`?
│   │   └── No → Check .mcp.json or ~/.claude.json
│   ├── Run with `--mcp-debug`
│   ├── Verify API keys/tokens
│   ├── Check timeout settings
│   └── Test independently with JSON-RPC
│
└── Performance issues?
    ├── Run `/doctor` for diagnostics
    ├── Check `/context` for usage
    ├── Review MCP server count
    └── Clean ~/.claude.json if startup slow
```

### Recovery Procedures

```bash
# Full reset
exit
rm ~/.claude/sessions/*.json
claude  # Fresh start

# Resume after crash
claude --resume  # Resume last session

# Debug MCP issues
claude --mcp-debug

# Verify JSON config
cat ~/.claude.json | python -m json.tool
```

---

## Version History

- v1.0.0 (2024-12): Initial release based on ecosystem analysis
