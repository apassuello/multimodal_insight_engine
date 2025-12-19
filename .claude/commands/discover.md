---
description: Find and evaluate Claude Code extensions (skills, commands, hooks, MCP servers, agents, plugins) for a specific need
allowed-tools: Read, Glob
argument-hint: [what capability or problem you need to solve]
---

# Discover Extensions

Read `.claude/docs/resource-discovery-reference.md` to understand available resources and evaluation criteria.

## User Need
$ARGUMENTS

## Your Task

1. **Identify the need type**: What kind of capability is being requested?
   - Domain expertise (automatic) → Skill
   - Explicit user action → Command  
   - Guaranteed enforcement → Hook
   - External service access → MCP Server
   - Complex orchestration → Agent

2. **Search the reference**: Find relevant resources in the discovery reference that match the need.

3. **Evaluate options**: Apply the quality criteria (stars, maintenance, documentation) to any recommendations.

4. **Recommend**: Provide 1-3 options ranked by fit, with brief rationale for each.

## Output Format

```
## Recommended: [Resource Type]

### Option 1: [Name]
- Source: [Repository or location]
- Why: [Brief rationale for this recommendation]
- Setup: [One-line setup hint]

### Option 2: [Name] (if applicable)
...

## Alternative Approaches
[If the need could be met differently, mention briefly]

## Next Steps
[What the user should do to proceed]
```