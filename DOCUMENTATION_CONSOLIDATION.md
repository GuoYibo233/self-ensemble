# Documentation Consolidation Plan

This document outlines the consolidation of documentation files to reduce redundancy and improve clarity.

## Current Documentation Issues

1. **Too many root-level markdown files** (9 files)
2. **Duplicate content** across multiple files
3. **Outdated information** in some files
4. **Unclear organization** - hard to find specific information

## Consolidation Strategy

### Files to Merge/Remove

#### 1. Changelog Files (merge into CHANGELOG.md)
- ✅ Keep: `CHANGELOG.md` (main changelog)
- ❌ Remove: `RECENT_UPDATES.md` (content merged into CHANGELOG.md)
- ❌ Remove: `IMPLEMENTATION_SUMMARY.md` (content merged into CHANGELOG.md)

#### 2. Improvement Documentation (merge into docs/)
- ❌ Remove: `IMPROVEMENTS_SUMMARY.md` (move to docs/IMPROVEMENTS.md)
- ❌ Remove: `BEFORE_AFTER_COMPARISON.md` (move to docs/IMPROVEMENTS.md)
- ❌ Remove: `CHANGES_README.md` (content merged into README.md and docs/)

#### 3. Debug/Index Files (merge into docs/)
- ❌ Remove: `DEBUG_INDEX.md` (content merged into docs/DELEGATE_PROMPT.md)

#### 4. FlexAttention Documentation (already in docs/)
- ✅ Keep: `FLEXATTENTION_USAGE.md` (comprehensive usage guide)
- ✅ Keep: docs/README_FLEXATTENTION.md
- ✅ Keep: docs/FLEX_ATTENTION_IMPLEMENTATION.md

### Final Structure

```
Root Level (4 files only):
├── README.md                    # Main entry point
├── CHANGELOG.md                 # All changes and updates
├── FLEXATTENTION_USAGE.md       # Comprehensive usage guide
└── BASELINE_USAGE.md            # NEW: Baseline generation guide

docs/ directory (organized by topic):
├── QUICKSTART.md               # Quick start guide
├── ARCHITECTURE.md             # Architecture overview
├── DELEGATE_PROMPT.md          # Debugging guide
├── LINUX_SETUP.md              # Setup instructions
├── README_FLEXATTENTION.md     # FlexAttention overview
├── FLEX_ATTENTION_IMPLEMENTATION.md  # Technical details
├── CREATE_FLEX_ATTENTION_MASK_IMPLEMENTATION.md
├── QUICK_REFERENCE.md          # API reference
├── REUSE_VS_NEW_DETAILED.md    # Component breakdown
├── IMPROVEMENTS.md             # NEW: Consolidated improvements
├── SETUP_VERIFICATION.md       # Verification guide
├── 实现总结.md                 # Chinese summary
└── [Internal prompts...]       # Keep for reference
```

## Consolidation Actions

### Action 1: Create BASELINE_USAGE.md
Consolidate baseline-related content from multiple sources into a single comprehensive guide.

### Action 2: Create docs/IMPROVEMENTS.md
Merge IMPROVEMENTS_SUMMARY.md, BEFORE_AFTER_COMPARISON.md, and CHANGES_README.md.

### Action 3: Update CHANGELOG.md
Merge content from RECENT_UPDATES.md and IMPLEMENTATION_SUMMARY.md.

### Action 4: Update README.md
- Add baseline generation section
- Update documentation links
- Simplify structure

### Action 5: Remove Redundant Files
After merging content, remove:
- RECENT_UPDATES.md
- IMPLEMENTATION_SUMMARY.md
- IMPROVEMENTS_SUMMARY.md
- BEFORE_AFTER_COMPARISON.md
- CHANGES_README.md
- DEBUG_INDEX.md

## Benefits

1. **Clarity**: Clear separation between user docs and internal docs
2. **Maintainability**: Fewer files to keep in sync
3. **Discoverability**: Easier to find relevant information
4. **Up-to-date**: Single source of truth for each topic
5. **Organization**: Logical grouping in docs/ directory
