# MultiModal Insight Engine Guidelines

## Build, Test, & Lint Commands
```bash
# Run all tests with coverage
./run_tests.sh

# Run a single test file
python -m pytest tests/test_file.py -v

# Run a specific test function
python -m pytest tests/test_file.py::test_function -v

# Run lint checks
flake8 src/ tests/

# Type checking
mypy src/ tests/
```

## Style Guidelines
- **Code Style**: Follow PEP 8 (4-space indentation, 79-char line limit)
- **Imports**: Group standard lib → third-party → local; alphabetically sorted in each group
- **Typing**: Use type hints for all function parameters and returns
- **Docstrings**: Google-style docstrings with Args/Returns sections
- **Classes**: CamelCase for classes, snake_case for functions/variables
- **Error Handling**: Validate inputs with assertions, use specific exception types
- **Testing**: Every module should have corresponding tests in tests/ directory
- **Module Headers**: Include module-level docstrings with PURPOSE and KEY COMPONENTS
- **Naming**: Descriptive variable names that indicate purpose and type

## Project Priorities
- **Test-Driven Development**: We follow TDD practices with comprehensive test coverage
- **Security**: Security and constitutional AI principles are critical for this ML system
- **Documentation**: Maintain comprehensive technical documentation and architecture guides
- **Code Quality**: Enforce strict code review standards and refactoring best practices
- **Performance**: ML model performance and optimization are key considerations

## Available Specialized Agents

Agents are specialized experts available in `.claude/agents/`. They automatically activate based on task context or can be explicitly invoked.

### Documentation & Architecture
When working on documentation, architecture diagrams, or technical guides:
- Technical documentation and architecture analysis
- API documentation and OpenAPI specifications
- Tutorial creation and educational content
- Diagrams (Mermaid flowcharts, ERDs, architecture)
- Reference documentation generation

### Code Quality & Review
For code review, architectural decisions, and quality assurance:
- Comprehensive code review with AI-powered analysis
- Architecture pattern validation and design review
- Security audits and vulnerability assessment
- Legacy code modernization and refactoring

### Testing & Quality Assurance
When implementing tests, following TDD, or debugging:
- TDD orchestration and test-driven development workflows
- AI-powered test automation and test generation
- Mutation testing and test quality validation
- Performance testing and load testing
- Integration and contract testing
- Chaos engineering and resilience testing
- Test data generation and management

### Debugging & Developer Experience
For error analysis, debugging, and developer tooling:
- Root cause analysis and error debugging
- Log analysis and error pattern detection
- Developer experience optimization

### Deployment & Infrastructure
For deployment pipelines, infrastructure, and DevOps:
- CI/CD pipeline design and deployment automation
- Infrastructure as Code (Terraform) expertise
- Project health auditing and monitoring

## Available Skills

Skills in `.claude/skills/` activate automatically when relevant patterns are detected:

### ML/AI Development
- RAG implementation patterns
- ML pipeline workflows and MLOps
- LLM evaluation strategies
- Prompt engineering patterns
- LangChain architecture

### Testing Automation
- API test automation
- Mutation testing
- Performance test suites
- Integration testing
- Contract validation
- Chaos engineering
- Test data generation

## Agent Usage Guidelines

**When to invoke agents:**
- **Explicitly**: Request by name for specialized tasks ("Use security-auditor to review this code")
- **Automatically**: Mention task type naturally ("Review this for security issues" triggers security-auditor)
- **Parallel**: Complex analysis can use multiple agents simultaneously

**Best practices:**
- Use specific agents for deep focused work (architecture review, security audit)
- Skills apply automatically for patterns (error handling, test generation)
- Combine agents for comprehensive analysis (code-reviewer + security-auditor + architect-review)