# GitHub MCP Integration Guide

> **Purpose**: Documentation for GitHub MCP server capabilities and usage
> **Created**: 2025-12-19
> **Scope**: Local to this project (private to your account)

---

## Overview

The GitHub MCP (Model Context Protocol) server is now integrated with Claude Code, providing direct access to GitHub repositories, issues, pull requests, and workflows.

**Status**: ✓ Connected
**Scope**: Local (private to you in this project)
**Authentication**: Uses `GITHUB_PERSONAL_ACCESS_TOKEN` from environment

---

## Available Capabilities

### 🔍 **Repository Operations**

**Browse and search repositories:**
- List your repositories
- Search code across repos
- View repository contents
- Get file contents from any branch
- Search for files by name or path

**Examples:**
```
"Show me all my Python repositories"
"Search for 'transformer' in my repositories"
"Get the contents of README.md from my ml-project repo"
```

### 🐛 **Issue Management**

**Create and manage issues:**
- Create new issues
- List issues (open, closed, all)
- Update issue status
- Add comments to issues
- Search issues by labels, assignee, state

**Examples:**
```
"Create a GitHub issue titled 'Add Gradio demo' in my multimodal_insight_engine repo"
"Show me all open issues labeled 'bug' in this repository"
"Add a comment to issue #42 explaining the fix"
```

### 🔀 **Pull Request Workflows**

**PR operations:**
- Create pull requests
- List PRs by state (open, closed, merged)
- Review PR details
- Add comments to PRs
- Check PR status and reviews

**Examples:**
```
"Create a PR for the feature/gradio-demo branch"
"Show me all open PRs in this repository"
"What's the status of PR #15?"
```

### 📊 **Project Insights**

**Repository analysis:**
- View commit history
- Check branch information
- Get repository statistics
- View contributors
- Analyze code changes

**Examples:**
```
"Show me the last 10 commits on the main branch"
"What branches exist in this repository?"
"Who are the top contributors to this repo?"
```

### 🔔 **Notifications & Events**

**Stay updated:**
- View notifications
- Check repository events
- Track issue activity
- Monitor PR updates

---

## Common Workflows

### 1. **Portfolio Standardization**

```
"Analyze my GitHub repositories and create a standardized README template
that includes: project overview, installation, usage, demo links,
tech stack, and contribution guidelines"
```

### 2. **Issue-Driven Development**

```
"Create issues for the following tasks:
1. Implement Gradio demo interface
2. Add model performance benchmarks
3. Create deployment documentation
4. Set up GitHub Pages for demo"
```

### 3. **PR Management**

```
"Create a PR for my model optimization work with:
- Title: 'Optimize inference speed by 40%'
- Description with benchmark results
- Link to related issue #23"
```

### 4. **Code Discovery**

```
"Search my repositories for examples of FastAPI + ML model integration"
"Find all README files that mention 'transformer'"
```

### 5. **Repository Health Check**

```
"Analyze this repository and suggest improvements for:
- Documentation completeness
- Test coverage visibility
- GitHub Actions setup
- Project organization"
```

---

## Integration with ML Workflows

### **Automated Documentation**

```
# After training a model:
"Create a GitHub issue documenting today's experiment:
- Model: ResNet50
- Accuracy: 94.2%
- Training time: 2.5 hours
- Key hyperparameters: lr=0.001, batch=32"
```

### **Experiment Tracking**

```
# Track ML experiments via issues:
"Create an issue for experiment #42 with label 'experiment'
including the results from results.json"
```

### **Demo Deployment**

```
# After creating Gradio demo:
"Create a PR that adds the Gradio demo with:
- app.py implementation
- requirements.txt update
- README section for running the demo
- Screenshots in the PR description"
```

### **Model Release**

```
# Prepare model release:
"Create a GitHub release v1.0.0 with:
- Release notes from CHANGELOG.md
- Model checkpoint (link to HuggingFace)
- Performance benchmarks
- Usage instructions"
```

---

## Security & Permissions

### **Token Scope**

Your `GITHUB_PERSONAL_ACCESS_TOKEN` requires these scopes:
- ✅ `repo` - Full repository access
- ✅ `read:org` - Read organization membership
- ✅ `workflow` - Update GitHub Actions workflows (optional)

### **Best Practices**

✅ **DO:**
- Use personal access tokens with minimum required scopes
- Rotate tokens regularly (every 90 days)
- Use environment variables (never hardcode)
- Review MCP server logs for unusual activity

❌ **DON'T:**
- Share your token or commit it to repos
- Grant excessive permissions
- Use the same token across multiple machines
- Allow third-party access to MCP server

### **Token Management**

**Create a new token:**
1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Generate new token (classic)
3. Select scopes: `repo`, `read:org`
4. Copy token and set in environment:
   ```bash
   export GITHUB_PERSONAL_ACCESS_TOKEN="ghp_..."
   ```

**Revoke token:**
```bash
# Remove from Claude:
claude mcp remove github -s local

# Revoke on GitHub:
# Settings → Developer settings → Personal access tokens → Revoke
```

---

## Troubleshooting

### **Connection Issues**

**Symptom**: "MCP server not connected"

**Solutions:**
1. Verify token is set: `echo $GITHUB_PERSONAL_ACCESS_TOKEN | head -c 10`
2. Check token permissions on GitHub
3. Restart Claude Code
4. Check MCP logs: `claude mcp list --verbose`

### **Permission Errors**

**Symptom**: "403 Forbidden" or "404 Not Found"

**Solutions:**
1. Verify token has correct scopes
2. Check repository access (private repos need token access)
3. Confirm organization permissions
4. Re-create token with proper scopes

### **Rate Limiting**

**Symptom**: "API rate limit exceeded"

**Solutions:**
1. Wait for rate limit reset (check response headers)
2. Authenticated requests have higher limits (5000/hour vs 60/hour)
3. Use GraphQL API for complex queries (counts as 1 request)
4. Cache results when possible

---

## Performance Tips

### **Minimize API Calls**

❌ **Bad**:
```
"Get file1.py, then get file2.py, then get file3.py..."
```

✅ **Good**:
```
"Get the contents of file1.py, file2.py, and file3.py from the repo"
```

### **Use Specific Queries**

❌ **Bad**:
```
"Show me everything about this repository"
```

✅ **Good**:
```
"Show me the README, last 5 commits, and open issues labeled 'bug'"
```

### **Batch Operations**

Use GitHub's batch APIs when available through the MCP server for creating multiple issues, updating labels, etc.

---

## Advanced Usage

### **GraphQL Queries**

The GitHub MCP server supports GraphQL for complex queries:

```
"Use GitHub GraphQL to find all repositories with:
- More than 10 stars
- Written in Python
- Updated in the last month
- With open issues"
```

### **Webhooks Integration**

While MCP doesn't directly handle webhooks, you can:
1. Check webhook events via API
2. List recent deliveries
3. Analyze webhook payloads

### **GitHub Actions Integration**

```
"List all GitHub Actions workflows in this repository"
"Show me the status of the last CI/CD run"
"Get the logs from the failed test workflow"
```

---

## MCP Server Details

**Configuration location**: `~/.claude.json` (under this project path)

**View configuration:**
```bash
claude mcp get github
```

**Remove server:**
```bash
claude mcp remove github -s local
```

**Update server:**
```bash
# Remove old version
claude mcp remove github -s local

# Add new version
claude mcp add --transport stdio github \
  --env GITHUB_PERSONAL_ACCESS_TOKEN=$GITHUB_TOKEN \
  -- npx -y @modelcontextprotocol/server-github
```

---

## Integration with Other Tools

### **With Qdrant MCP** (Already installed)

```
"Search my knowledge base for ML techniques, then create GitHub issues
for implementing the top 3 most relevant approaches"
```

### **With ML Plugins**

```
"After training completes, create a GitHub issue with:
- Model metrics from experiment-tracking
- Visualizations from data-visualization-creator
- Deployment notes from model-deployment-helper"
```

### **With Frontend Skills**

```
"Create a GitHub Pages site using web-artifacts-builder showcasing:
- My ML project portfolio
- Interactive demos
- Model performance benchmarks
Then create a PR to enable GitHub Pages"
```

---

## Example Workflows

### **Setup New ML Project on GitHub**

```
1. "Create a new GitHub repository named 'ml-project-name'"
2. "Create initial issues for: data collection, model training, evaluation, deployment"
3. "Create a project board with columns: To Do, In Progress, Done"
4. "Add these issues to the project board"
```

### **Weekly Progress Report**

```
"Generate a progress report for this week:
- List all closed issues
- Show merged PRs
- Summarize commit activity
- Highlight key changes"
```

### **Pre-Release Checklist**

```
"Before release v1.0:
- List all open issues labeled 'blocker'
- Check if all tests are passing (GitHub Actions)
- Verify README is up to date
- Create draft release notes from closed issues"
```

---

## Resources

- **GitHub MCP Server**: [@modelcontextprotocol/server-github](https://github.com/modelcontextprotocol/servers/tree/main/src/github)
- **GitHub API Docs**: [docs.github.com/rest](https://docs.github.com/rest)
- **Token Management**: [GitHub Settings](https://github.com/settings/tokens)
- **MCP Protocol**: [modelcontextprotocol.io](https://modelcontextprotocol.io)

---

## Version History

- v1.0.0 (2025-12-19): Initial GitHub MCP integration
  - Configured for local project scope
  - Using GITHUB_PERSONAL_ACCESS_TOKEN from environment
  - Full repository, issue, and PR access enabled
