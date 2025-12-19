# GitHub MCP Server Setup Guide

## Overview

The GitHub MCP server provides integration with GitHub for:
- Checking CI/CD workflow status
- Managing pull requests
- Reviewing GitHub Actions runs
- Accessing repository information

## Prerequisites

- GitHub personal access token (classic or fine-grained)
- Claude Code with MCP support

## Setup Steps

### 1. Create GitHub Personal Access Token

**Option A: Fine-Grained Token (Recommended)**

1. Go to GitHub Settings → Developer settings → Personal access tokens → Fine-grained tokens
2. Click "Generate new token"
3. Configure:
   - **Token name**: "Claude Code MCP"
   - **Expiration**: 90 days (or custom)
   - **Repository access**: Select "apassuello/constitutional-ai"
   - **Permissions**:
     - Actions: Read
     - Contents: Read
     - Pull requests: Read and write
     - Workflows: Read and write

**Option B: Classic Token**

1. Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click "Generate new token (classic)"
3. Select scopes:
   - `repo` (Full control of private repositories)
   - `workflow` (Update GitHub Action workflows)
4. Generate and copy token

### 2. Configure MCP Server in Claude Code

Add to your Claude Code settings (usually `~/.config/claude/settings.json` or via Claude Code settings UI):

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_your_token_here"
      }
    }
  }
}
```

**Alternative: Environment Variable**

Or set environment variable before starting Claude Code:

```bash
export GITHUB_PERSONAL_ACCESS_TOKEN="ghp_your_token_here"
```

### 3. Verify Installation

After configuring, restart Claude Code and check that the GitHub MCP server is available:

```
Ask Claude: "Can you check the GitHub MCP server status?"
```

Claude should be able to access GitHub tools like:
- List repositories
- Check workflow runs
- Get pull request information
- View commit history

## Usage Examples

### Check CI Status

```
"Check the status of recent CI runs on constitutional-ai"
```

### Review PR

```
"Show me the details of PR #123"
```

### List Recent Workflows

```
"List recent GitHub Actions workflow runs"
```

### Check Test Results

```
"Did the latest CI run pass all tests?"
```

## Troubleshooting

### Token Permission Issues

If you see permission errors:
1. Verify token has required scopes/permissions
2. Check repository access settings
3. Regenerate token if needed

### MCP Server Not Starting

If GitHub MCP server doesn't start:
1. Ensure `npx` is available: `npx --version`
2. Check Claude Code logs for errors
3. Verify JSON configuration syntax
4. Restart Claude Code

### Rate Limiting

GitHub API has rate limits:
- **Authenticated**: 5,000 requests/hour
- **Unauthenticated**: 60 requests/hour

With token, you should have sufficient quota for normal usage.

## Security Best Practices

1. **Never commit tokens** - Add to `.gitignore` if storing in files
2. **Use minimal permissions** - Fine-grained tokens with repository-specific access
3. **Set expiration** - Rotate tokens periodically (90 days recommended)
4. **Revoke if compromised** - Immediately revoke in GitHub settings
5. **Store securely** - Use environment variables or secure credential storage

## Alternative: GitHub CLI (gh)

If you already have GitHub CLI (`gh`) authenticated, you can use it via Bash tool:

```bash
# Check workflow status
gh run list --repo apassuello/constitutional-ai

# View specific run
gh run view <run-id> --repo apassuello/constitutional-ai

# Check PR status
gh pr view <pr-number> --repo apassuello/constitutional-ai
```

This doesn't require MCP setup but is more manual.

## Benefits of MCP vs CLI

| Feature | MCP Server | GitHub CLI |
|---------|-----------|------------|
| **Automatic tool discovery** | ✅ Claude knows available tools | ❌ Manual commands |
| **Structured data** | ✅ Parsed into objects | ⚠️ Text output |
| **Error handling** | ✅ Built-in | ⚠️ Manual parsing |
| **Setup complexity** | ⚠️ Requires token config | ✅ One-time `gh auth login` |
| **Usage** | ✅ Natural language | ⚠️ Remember commands |

## Current Project CI/CD

Your constitutional-ai project has 3 GitHub Actions workflows:

1. **ci.yml** - Tests on Python 3.10, 3.11, 3.12
2. **code-quality.yml** - Black, isort, flake8, ruff, mypy
3. **dependency-review.yml** - Security scanning

With GitHub MCP, Claude can:
- Check if tests are passing
- View linting results
- Monitor security scans
- Review PR checks before merging

## Resources

- [Official GitHub MCP Server](https://github.com/modelcontextprotocol/servers/tree/main/src/github)
- [MCP Documentation](https://modelcontextprotocol.io/)
- [GitHub API Documentation](https://docs.github.com/en/rest)
