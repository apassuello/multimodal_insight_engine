# Skill Optimization Guide

> **Generated**: 2025-12-19
> **Purpose**: Recommendations for optimizing long skills using progressive disclosure pattern

---

## Overview

Based on configuration audit, 5 skills exceed or approach the 400-line community best practice threshold. This guide provides specific optimization recommendations for each.

**Optimization Strategy**: Progressive Disclosure
- Keep SKILL.md focused on triggers, overview, and key patterns (~150-200 lines)
- Move detailed implementation examples to reference files in skill directory
- Link to references for "deep dive" content
- Reduces token usage when skill activates but maintains full functionality

---

## Skill-by-Skill Recommendations

### 1. llm-evaluation (471 lines → target: ~180 lines)

**Current structure:**
- Lines 1-63: Overview and core concepts ✅ Keep
- Lines 64-95: Quick start ✅ Keep (abbreviated)
- Lines 97-441: Detailed implementations ❌ Move to references
- Lines 443-471: Resources and best practices ✅ Keep

**Optimization plan:**
```
llm-evaluation/
├── SKILL.md (180 lines)
│   ├── When to use
│   ├── Core evaluation types (overview)
│   ├── Quick start (single example)
│   ├── Link to detailed implementations
│   └── Best practices
│
├── references/
│   ├── automated-metrics.md        ← Lines 97-182 (BLEU, ROUGE, BERTScore)
│   ├── llm-as-judge.md             ← Lines 184-245 (judge patterns)
│   ├── human-evaluation.md         ← Lines 247-309 (annotation, agreement)
│   ├── ab-testing.md               ← Lines 311-366 (statistical testing)
│   ├── regression-testing.md       ← Lines 368-403 (regression detection)
│   └── benchmarking.md             ← Lines 405-441 (benchmark runner)
```

**Expected savings**: ~290 lines moved to references (loaded only when needed)

---

### 2. rag-implementation (403 lines → target: ~170 lines)

**Current structure:**
- Lines 1-59: Overview and components ✅ Keep
- Lines 60-98: Quick start ✅ Keep
- Lines 100-373: Advanced patterns and implementations ❌ Move to references
- Lines 375-403: Resources and best practices ✅ Keep

**Optimization plan:**
```
rag-implementation/
├── SKILL.md (170 lines)
│   ├── When to use
│   ├── Core components (overview)
│   ├── Quick start (basic example)
│   ├── Link to advanced patterns
│   └── Best practices
│
├── references/
│   ├── advanced-patterns.md        ← Lines 100-168 (hybrid, multi-query, compression)
│   ├── chunking-strategies.md      ← Lines 170-215 (splitters)
│   ├── vector-stores.md            ← Lines 217-250 (Pinecone, Weaviate, Chroma)
│   ├── retrieval-optimization.md   ← Lines 252-300 (filtering, MMR, reranking)
│   ├── prompt-engineering.md       ← Lines 302-339 (RAG prompts)
│   └── evaluation.md               ← Lines 341-373 (metrics)
```

**Expected savings**: ~230 lines moved to references

---

### 3. ml-type-hints (399 lines → target: ~200 lines)

**Current structure:**
- Lines 1-65: When to use, project status, guidelines ✅ Keep
- Lines 66-226: PyTorch patterns and examples ⚠️ Consolidate
- Lines 227-360: Known mypy challenges (project-specific) ✅ Keep (important context)
- Lines 334-399: Best practices and configuration ✅ Keep

**Optimization plan:**
```
ml-type-hints/
├── SKILL.md (200 lines)
│   ├── When to use
│   ├── Project type checking status
│   ├── When to use/not use hints (keep guidelines)
│   ├── Quick reference (most common patterns only)
│   ├── Known mypy challenges (keep - project-specific)
│   ├── Best practices
│   └── Link to comprehensive patterns
│
├── references/
│   ├── pytorch-patterns.md         ← Comprehensive tensor, model, device types
│   ├── transformers-patterns.md    ← HuggingFace-specific patterns
│   └── advanced-patterns.md        ← Generator types, protocol types, complex cases
```

**Expected savings**: ~200 lines consolidated, maintain critical project context

**Note**: This skill is more reference-heavy due to project-specific mypy errors. Line count is acceptable given the value.

---

### 4. pytest-testing (368 lines → target: ~180 lines)

**Current structure:**
- Lines 1-59: When to use, naming conventions ✅ Keep
- Lines 60-178: Pytest patterns with examples ⚠️ Consolidate
- Lines 179-209: Coverage and organization ✅ Keep
- Lines 210-292: Running tests, quality checklist ✅ Keep (concise reference)

**Optimization plan:**
```
pytest-testing/
├── SKILL.md (180 lines)
│   ├── When to use
│   ├── Project conventions (naming, structure)
│   ├── Quick patterns (most common 3-4 patterns)
│   ├── Coverage guidelines
│   ├── Running tests
│   ├── Quality checklist
│   └── Link to comprehensive patterns
│
├── references/
│   ├── pytest-patterns.md          ← All fixture, parametrize, exception patterns
│   ├── mocking-guide.md            ← AI component mocking strategies
│   └── advanced-testing.md         ← Complex scenarios, edge cases
```

**Expected savings**: ~190 lines moved to references

---

### 5. langchain-architecture (338 lines → monitor, acceptable)

**Current status**: 338 lines (under 400-line threshold)

**Assessment**:
- Well-structured with clear sections
- Good balance of overview and examples
- No immediate optimization needed

**Recommendation**: Monitor - if grows beyond 400 lines, apply similar pattern:
- Move detailed code examples to references
- Keep architecture patterns and decision frameworks in main skill

---

## Implementation Priority

### Phase 1 (High Impact)
1. **llm-evaluation** (471 → 180 lines, 62% reduction)
2. **rag-implementation** (403 → 170 lines, 58% reduction)

**Impact**: ~520 line reduction in most frequently used skills

### Phase 2 (Moderate Impact)
3. **pytest-testing** (368 → 180 lines, 51% reduction)

**Impact**: ~190 line reduction, commonly used in development

### Phase 3 (Optimization)
4. **ml-type-hints** (399 → 200 lines, 50% reduction while preserving project context)

**Impact**: ~200 line reduction, but skill is already well-contextualized

### Monitor
5. **langchain-architecture** - acceptable at 338 lines

---

## Progressive Disclosure Template

When refactoring skills, use this pattern:

```markdown
---
name: skill-name
description: Trigger conditions...
---

# Skill Title

## When to Use This Skill
[Clear trigger conditions]

## Core Concepts
[High-level overview - what, not how]

## Quick Start
[Single, minimal working example]

## Key Patterns
[2-3 most common patterns with brief examples]

## Advanced Topics
For detailed implementations, see:
- [Topic 1](references/topic1.md) - Description
- [Topic 2](references/topic2.md) - Description

## Best Practices
[Concise bullet points]

## Common Issues
[Brief troubleshooting]
```

---

## Benefits of Optimization

### Token Efficiency
- **Before**: 471-line skill loads entirely when activated (~1,500 tokens)
- **After**: 180-line skill loads + references only if explicitly requested (~600 tokens base)
- **Savings**: ~60% reduction in standard activation, 100% availability when needed

### Maintenance
- Easier to update specific patterns without scrolling through large files
- Clear separation between triggers/overview vs. implementation details
- Easier for team members to contribute specific patterns

### Activation Accuracy
- Shorter, focused descriptions improve Claude's ability to determine relevance
- Less noise in skill metadata reduces false activations

---

## Next Steps

1. **Backup current skills** before refactoring
2. **Start with Phase 1** (llm-evaluation, rag-implementation)
3. **Test activation** after refactoring - ensure skills still trigger appropriately
4. **Measure impact** - check if skills activate correctly and reduce token usage
5. **Iterate** based on results before proceeding to Phase 2

---

## Validation Checklist

After optimizing a skill:

- [ ] Skill still activates in appropriate contexts
- [ ] Quick start example remains functional
- [ ] Reference files are properly linked
- [ ] All original content is preserved (just relocated)
- [ ] Description triggers are clear and specific
- [ ] Best practices section remains comprehensive
- [ ] Line count reduced by at least 40%

---

## Version History

- v1.0.0 (2024-12-19): Initial optimization recommendations based on audit
