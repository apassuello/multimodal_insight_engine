# MCP Server Configuration Guide

> **Purpose**: Understanding MCP server scopes and configuration for this project
> **Created**: 2025-12-19
> **Current Status**: Only GitHub MCP enabled for this project

---

## Current Configuration

### **Active MCP Servers**

**This Project:**
- ✅ **GitHub** (project-specific via `.mcp.json`)

**User-Level** (applies to all projects):
- None (cleaned up for project isolation)

---

## MCP Server Scopes Explained

### **Three Configuration Levels**

| Level | File Location | Scope | Use Case |
|-------|--------------|-------|----------|
| **Project** | `.mcp.json` | This project only | Project-specific integrations |
| **Local** | `~/.claude.json` (under project path) | Private to you in this project | Personal overrides |
| **User** | `~/.claude.json` (top level) | All your projects | Tools you use everywhere |

### **Precedence & Conflicts**

- `.mcp.json` servers are **project-specific** (shared via git)
- User-level servers in `~/.claude.json` apply to **all projects**
- Both sets of servers are **additive** (all are active)
- To make a project use ONLY specific servers, remove user-level servers

---

## Why Only GitHub for This Project?

### **Design Decision**

This ML project only needs GitHub integration for:
- Repository management
- Issue tracking and project boards
- Pull request workflows
- Code search and documentation
- Portfolio standardization

### **Other MCP Servers**

**Not needed here:**
- `md_cv_rag` - Removed (was for CV/resume project)
- `huggingface` - In different project path
- `playwright` - In different project path

**Could add later:**
- PostgreSQL MCP - If you add a database
- AWS MCP - For cloud deployment
- Slack MCP - For team notifications

---

## Project-Specific Configuration (.mcp.json)

### **Current File**

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"
      }
    }
  }
}
```

### **Benefits of .mcp.json**

✅ **Version controlled** - Team gets the same MCP servers
✅ **Project-specific** - Only what this project needs
✅ **Portable** - Works on any team member's machine
✅ **Documented** - Clear what integrations are used

### **Git Tracking**

The `.mcp.json` file **should be committed** to git because:
- Team members need the same integrations
- Documents project dependencies
- Ensures consistent experience
- No secrets (uses environment variables)

---

## Adding More MCP Servers

### **For This Project Only**

Edit `.mcp.json` and add to the `mcpServers` object:

```json
{
  "mcpServers": {
    "github": { /* existing */ },
    "postgres": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres"],
      "env": {
        "POSTGRES_URL": "${DATABASE_URL}"
      }
    }
  }
}
```

### **For All Your Projects**

Use the CLI to add to user level:

```bash
claude mcp add --transport stdio server-name \
  --env VAR_NAME=value \
  -- command args
```

---

## Common MCP Servers for ML Projects

### **Databases**

```bash
# PostgreSQL (for experiment tracking)
claude mcp add --transport stdio postgres \
  --env POSTGRES_URL=$DATABASE_URL \
  -- npx -y @modelcontextprotocol/server-postgres

# MongoDB (for document storage)
claude mcp add --transport stdio mongodb \
  --env MONGODB_URI=$MONGO_URI \
  -- npx -y @modelcontextprotocol/server-mongodb
```

### **Cloud Services**

```bash
# AWS (for S3, SageMaker, etc.)
claude mcp add --transport stdio aws \
  --env AWS_ACCESS_KEY_ID=$AWS_KEY \
  --env AWS_SECRET_ACCESS_KEY=$AWS_SECRET \
  -- npx -y @modelcontextprotocol/server-aws

# Google Cloud
claude mcp add --transport stdio gcp \
  --env GOOGLE_APPLICATION_CREDENTIALS=$GCP_CREDS \
  -- npx -y @modelcontextprotocol/server-gcp
```

### **ML Platforms**

```bash
# Hugging Face (already in another project)
# Weight & Biases (for experiment tracking)
# MLflow (for model registry)
```

### **Communication**

```bash
# Slack (for notifications)
claude mcp add --transport stdio slack \
  --env SLACK_BOT_TOKEN=$SLACK_TOKEN \
  -- npx -y @modelcontextprotocol/server-slack
```

---

## Troubleshooting

### **"Why do I see servers I didn't add?"**

**Cause**: User-level servers in `~/.claude.json` apply to all projects

**Solution**:
```bash
# List all servers
claude mcp list

# Remove unwanted user-level servers
claude mcp remove server-name
```

### **"I want different MCP servers per project"**

**Solution**: Use `.mcp.json` for project-specific servers, keep user-level minimal

**Example**:
- Project A: GitHub + PostgreSQL (in `.mcp.json`)
- Project B: GitHub + MongoDB (in `.mcp.json`)
- User-level: None (or only universal tools)

### **"Team members can't connect to MCP servers"**

**Checklist**:
- [ ] `.mcp.json` is committed to git
- [ ] Environment variables are documented (README or .env.example)
- [ ] Team members have set required env vars
- [ ] Node.js/npx is installed
- [ ] Team members ran `npm install` if needed

### **"MCP server shows 'Not Connected'"**

**Debug steps**:
```bash
# Check server status
claude mcp list

# View detailed config
claude mcp get server-name

# Test with debug mode
claude --mcp-debug

# Verify environment variable
echo $GITHUB_PERSONAL_ACCESS_TOKEN | head -c 10
```

---

## Best Practices

### **✅ DO**

- Use `.mcp.json` for project-specific servers (git-tracked)
- Use environment variables for credentials (never hardcode)
- Document required env vars in README
- Keep user-level servers minimal
- Test MCP servers after adding (`claude mcp list`)

### **❌ DON'T**

- Hardcode credentials in `.mcp.json`
- Add every possible MCP server "just in case"
- Use user-level for project-specific needs
- Forget to document MCP requirements for team

---

## Security Considerations

### **Environment Variables**

**Good** (in `.mcp.json`):
```json
"env": {
  "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"
}
```

**Bad** (hardcoded):
```json
"env": {
  "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_1234567890abcdef"  // NEVER DO THIS!
}
```

### **Credentials Management**

**For this project**:
1. Set in shell profile: `export GITHUB_TOKEN="ghp_..."`
2. Or use `.env` file (gitignored): `GITHUB_TOKEN=ghp_...`
3. Load in shell: `source .env`
4. MCP uses `${GITHUB_TOKEN}` automatically

**Document in README**:
```markdown
## Required Environment Variables

- `GITHUB_TOKEN` - GitHub Personal Access Token with `repo` scope
  - Create at: https://github.com/settings/tokens
  - Required for MCP GitHub integration
```

---

## Monitoring & Health

### **Check Server Status**

```bash
# Quick health check
claude mcp list

# Detailed configuration
claude mcp get github

# View logs (if issues)
claude --mcp-debug
```

### **Performance Impact**

**Current**: 1 MCP server (GitHub)
- **Token cost**: ~2,000-5,000 tokens for tool definitions
- **Startup time**: +1-2 seconds
- **Status**: Minimal impact ✅

**If you add 5+ servers**:
- **Token cost**: Can reach 20,000+ tokens (10% of context)
- **Startup time**: +5-10 seconds
- **Status**: Consider selective enabling

### **Optimization**

Use `npx mcpick` to toggle servers when you have many:
```bash
# Install mcpick
npm install -g mcpick

# Toggle MCP servers interactively
npx mcpick
```

---

## Integration with Project

### **With GitHub Workflows**

The GitHub MCP enables:
- Creating issues from ML experiment results
- Automated PR creation for model updates
- Project board management for ML pipeline
- Code search for similar implementations

### **With ML Plugins**

Combined with your installed ML plugins:
```
1. Train model (ml-model-trainer plugin)
2. Evaluate results (model-evaluation-suite plugin)
3. Create GitHub issue documenting experiment (GitHub MCP)
4. Generate visualization (data-visualization-creator plugin)
5. Create PR with results (GitHub MCP)
```

### **With Frontend Skills**

```
1. Create Gradio demo (using frontend-design skill)
2. Push to repository (GitHub MCP)
3. Create GitHub Pages site (GitHub MCP)
4. Share portfolio link
```

---

## Version History

- v1.0.0 (2025-12-19): Initial MCP configuration
  - GitHub MCP server configured (project-specific)
  - Removed user-level servers for project isolation
  - Documented configuration patterns and best practices

---

## Resources

- **MCP Documentation**: [modelcontextprotocol.io](https://modelcontextprotocol.io)
- **Available MCP Servers**: [awesome-mcp-servers](https://github.com/punkpeye/awesome-mcp-servers)
- **GitHub MCP Server**: [@modelcontextprotocol/server-github](https://github.com/modelcontextprotocol/servers/tree/main/src/github)
- **Claude MCP Guide**: `.claude/docs/config-management-reference.md`
