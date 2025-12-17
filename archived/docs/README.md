# Documentation Index

This directory contains all documentation for the self-ensemble project, organized by category.

## Directory Structure

```
docs/
├── usage/              # Usage guides and tutorials
├── summaries/          # Project and task summaries
├── QUICKSTART.md       # Quick start guide (5 minutes)
├── ARCHITECTURE.md     # Architecture and design diagrams
├── IMPLEMENTATION_SUMMARY.md  # Implementation overview
├── CHANGELOG.md        # Chronological change log
└── [other technical docs]
```

## Getting Started

### New Users
1. Start here: **[QUICKSTART.md](QUICKSTART.md)** (5-minute setup)
2. Choose your method:
   - Baseline: [usage/BASELINE_USAGE.md](usage/BASELINE_USAGE.md)
   - FlexAttention: [usage/FLEXATTENTION_USAGE.md](usage/FLEXATTENTION_USAGE.md)
3. Check API reference: **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)**

### Developers
1. Read: **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)**
2. Review: **[ARCHITECTURE.md](ARCHITECTURE.md)**
3. Explore: **[FLEX_ATTENTION_IMPLEMENTATION.md](FLEX_ATTENTION_IMPLEMENTATION.md)**

## Documentation Categories

### 📖 Usage Guides ([usage/](usage/))
Detailed guides for using different features:
- [BASELINE_USAGE.md](usage/BASELINE_USAGE.md) - Baseline generation
- [FLEXATTENTION_USAGE.md](usage/FLEXATTENTION_USAGE.md) - FlexAttention methods
- [MYRIADLAMA_FLEX_USAGE.md](usage/MYRIADLAMA_FLEX_USAGE.md) - MyriadLAMA-specific usage
- [QUICK_BASELINE_GUIDE.md](usage/QUICK_BASELINE_GUIDE.md) - Quick baseline generation

### 📊 Project Summaries ([summaries/](summaries/))
Historical summaries and task completion records:
- [REORGANIZATION_SUMMARY.md](summaries/REORGANIZATION_SUMMARY.md) - Repository reorganization
- [MYRIADLAMA_IMPLEMENTATION_SUMMARY.md](summaries/MYRIADLAMA_IMPLEMENTATION_SUMMARY.md) - MyriadLAMA implementation
- [ENHANCEMENT_SUMMARY.md](summaries/ENHANCEMENT_SUMMARY.md) - Feature enhancements
- [TASK_COMPLETION_SUMMARY.md](summaries/TASK_COMPLETION_SUMMARY.md) - Task completion logs

### 🚀 Quick Start & Setup
- **[QUICKSTART.md](QUICKSTART.md)** - 5-minute quick start guide
- **[LINUX_SETUP.md](LINUX_SETUP.md)** - Linux-specific setup (Ubuntu 22.04, RTX A6000)
- **[SETUP_VERIFICATION.md](SETUP_VERIFICATION.md)** - Verify your environment

### 📚 Technical Documentation
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Implementation overview
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Architecture diagrams and design
- **[FLEX_ATTENTION_IMPLEMENTATION.md](FLEX_ATTENTION_IMPLEMENTATION.md)** - FlexAttention technical details
- **[CREATE_FLEX_ATTENTION_MASK_IMPLEMENTATION.md](CREATE_FLEX_ATTENTION_MASK_IMPLEMENTATION.md)** - Mask implementation guide
- **[MYRIADLAMA_FLEX_ATTENTION.md](MYRIADLAMA_FLEX_ATTENTION.md)** - MyriadLAMA FlexAttention details

### 🔍 Reference
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - API quick reference
- **[README_FLEXATTENTION.md](README_FLEXATTENTION.md)** - FlexAttention overview
- **[REUSE_VS_NEW_DETAILED.md](REUSE_VS_NEW_DETAILED.md)** - Code reuse breakdown

### 🔧 Development & Debugging
- **[DELEGATE_PROMPT.md](DELEGATE_PROMPT.md)** - Complete debugging guide
- **[IMPROVEMENTS.md](IMPROVEMENTS.md)** - Consolidated improvements
- **[BUGFIX_DOCUMENTATION_GUIDE.md](BUGFIX_DOCUMENTATION_GUIDE.md)** - Bug fix documentation guidelines
- **[FLEXATTENTION_BUGFIX_LOG.md](FLEXATTENTION_BUGFIX_LOG.md)** - FlexAttention bug fix log

### 📝 Change Logs & History
- **[CHANGELOG.md](CHANGELOG.md)** - Chronological change log
- **[DOCUMENTATION_CONSOLIDATION.md](DOCUMENTATION_CONSOLIDATION.md)** - Documentation consolidation plan

### 🌐 Additional Resources
- **[FLEXATTENTION_DOCS_README.md](FLEXATTENTION_DOCS_README.md)** - FlexAttention docs overview
- **[实现总结.md](实现总结.md)** - Implementation summary (Chinese/English bilingual)
- **[GITHUB_COPILOT_REVIEW_PROMPT.md](GITHUB_COPILOT_REVIEW_PROMPT.md)** - Code review guidelines

## Quick Navigation

**I want to...**
- Get started quickly → [QUICKSTART.md](QUICKSTART.md)
- Generate baseline results → [usage/BASELINE_USAGE.md](usage/BASELINE_USAGE.md)
- Use FlexAttention → [usage/FLEXATTENTION_USAGE.md](usage/FLEXATTENTION_USAGE.md)
- Understand the architecture → [ARCHITECTURE.md](ARCHITECTURE.md)
- Debug issues → [DELEGATE_PROMPT.md](DELEGATE_PROMPT.md)
- See what changed → [CHANGELOG.md](CHANGELOG.md)
- Migrate to new structure → [../MIGRATION_GUIDE.md](../MIGRATION_GUIDE.md)

## Contributing

When adding new documentation:
1. Place in appropriate subdirectory (usage/, summaries/, or root)
2. Update this README.md index
3. Add cross-references to related docs
4. Follow existing documentation style

## License & Acknowledgments

See main [README.md](../README.md) for license and acknowledgments.
