# English-Only Visualizations with Question Labels

## Overview

This document describes the new English-only visualization toolkit (`demo_visualization_v2.py`) that addresses specific requirements:
- All diagram labels in English only
- Clear question labeling (Question1, Question2, etc.)
- Prompt structure diagrams showing component relationships

## Generated Visualizations

### 1. Prompt Structure Diagram (`prompt_structure.png`)

**Purpose**: Shows how a single prompt is structured

**Components**:
- **INSTRUCTION**: Task description box (blue)
  - Example: "Answer the question based on your knowledge."
- **FEW-SHOT EXAMPLES**: Demonstration Q&A pairs (orange)
  - Example: "Q: What is 2+2? A: 4"
- **QUESTION**: The actual query to answer (green)
  - Example: "Q: What is the capital of Germany? A:"

**Visual Flow**: Instruction → Few-shot → Question (top to bottom with arrows)

### 2. Concatenated Questions Diagram (`concatenated_questions.png`)

**Purpose**: Shows how 5 paraphrased questions are concatenated for FlexAttention

**Key Features**:
- Each question clearly labeled: **Question1**, **Question2**, **Question3**, **Question4**, **Question5**
- Each box shows: `[Instruction + Few-shot + QuestionN]`
- Token counts displayed on the right (~45-53 tokens)
- Red dashed `[SEP]` separators between questions
- Blue sidebar showing "Original Content (Encoding Phase)"
- Green box at bottom showing "Generated Answer (can attend to all questions)"

**Color Coding**:
- Different colors for each question (from matplotlib Set3 colormap)
- Makes it easy to distinguish between paraphrases

### 3. Attention Mask with Labels (`attention_mask_labeled.png`)

**Purpose**: Shows the attention pattern with clear question labels

**Key Features**:
- **Question labels on both axes**:
  - Top: Question1, Question2, Question3, Question4, Question5, Generated
  - Left: Q1, Q2, Q3, Q4, Q5, Gen
- **Color coding**:
  - Green = Can attend
  - White = Cannot attend
- **Visual markers**:
  - Red dashed lines = Segment boundaries
  - Blue dashed lines = Generation start
- **Pattern shown**:
  - Block diagonal (questions isolated during encoding)
  - Full attention for generated tokens (fusion)

**Configuration Box** (right side):
```
Configuration:
  • Questions: 5
  • Tokens per question: ~12
  • Separator tokens: 2
  • Original length: 68
  • Generated tokens: 5
  • Total length: 73

Attention Pattern:
  • Each question isolated during encoding
  • Generated tokens attend to all questions
  • Enables information fusion
```

### 4. English-Only Flowchart (`flowchart_english.png`)

**Purpose**: Processing pipeline with English-only labels

**Steps** (color-coded):
1. **Input Preparation** (light blue)
2. **Generate 5 Paraphrases (Question1...Question5)** (light blue)
3. **Concatenate with [SEP]** (orange)
4. **Track Segment Positions** (orange)
5. **Create FlexAttention Mask** (purple)
6. **Patch Model Layers** (purple)
7. **Encoding Phase (Segment Isolation)** (green)
8. **Generation Loop (Fusion Attention)** (green)
9. **Token Selection (argmax)** (green)
10. **Decode Output** (pink)

## Usage

### Generate All Visualizations

```bash
cd plot/
python3 demo_visualization_v2.py
```

**Output**: Creates `demo_outputs_v2/` directory with 4 PNG files

### Key Improvements Over Original

| Feature | Original (`demo_visualization.py`) | New (`demo_visualization_v2.py`) |
|---------|-----------------------------------|----------------------------------|
| **Language** | Bilingual (Chinese/English) | English only |
| **Question Labels** | Generic segment markers | Explicit Question1, Question2, etc. |
| **Prompt Structure** | Not shown | Detailed diagram included |
| **Component Relationship** | Implicit | Explicit (Instruction + Few-shot + Question) |
| **Visual Clarity** | Good | Excellent (clearer labels) |

## Understanding the Visualizations

### How Questions Are Structured

Each "Question" in the diagrams actually contains three components:

```
Question1 = [Instruction + Few-shot Examples + Question1]
Question2 = [Instruction + Few-shot Examples + Question2]
...
Question5 = [Instruction + Few-shot Examples + Question5]
```

The questions are **paraphrases** of each other - they ask the same thing in different ways.

### How Attention Works

**During Encoding** (positions 0-67 in the example):
- Question1 tokens can only attend to Question1 tokens
- Question2 tokens can only attend to Question2 tokens
- etc.
- This is **segment isolation**

**During Generation** (positions 68-72 in the example):
- Generated tokens can attend to ALL previous tokens
- This includes all 5 questions
- This is **fusion attention**

### Why This Matters

- **Isolation prevents information leakage** between paraphrases during encoding
- **Fusion enables combining information** from all paraphrases during generation
- Result: More robust and accurate answers

## File Locations

```
plot/
├── demo_visualization_v2.py          # New script (English-only)
├── demo_outputs_v2/
│   ├── prompt_structure.png          # NEW: Shows Instruction + Few-shot + Question
│   ├── concatenated_questions.png    # NEW: Shows Question1...Question5 labeled
│   ├── attention_mask_labeled.png    # NEW: Question labels on axes
│   └── flowchart_english.png         # NEW: English-only flowchart
└── demo_visualization.py             # Original (bilingual, kept for reference)
```

## Quick Reference

### Question Labeling Convention

- **Question1** = First paraphrase of the query
- **Question2** = Second paraphrase of the query
- **Question3** = Third paraphrase of the query
- **Question4** = Fourth paraphrase of the query
- **Question5** = Fifth paraphrase of the query
- **Generated** = Model's generated answer

### Color Convention

- **Light Blue** = Input phase
- **Orange** = Processing phase
- **Purple** = Attention setup
- **Green** = Generation phase / Can attend
- **Pink** = Output phase
- **White** = Cannot attend
- **Red dashed** = Segment boundaries
- **Blue dashed** = Generation start

## For Presentations/Papers

All images are generated at 150 DPI, suitable for:
- PowerPoint presentations
- LaTeX documents
- Web display

For higher resolution (300 DPI), modify the `dpi` parameter in `demo_visualization_v2.py`:

```python
fig.savefig(output_path, dpi=300, bbox_inches='tight')
```

## Summary

The new English-only visualizations provide:
✅ Clear question labeling throughout  
✅ Explicit prompt structure diagram  
✅ Better understanding of component relationships  
✅ Cleaner diagrams for international audiences  
✅ Professional quality for publications  

---

**Created**: 2024-10-28  
**Script**: `demo_visualization_v2.py`  
**Output**: `demo_outputs_v2/`
