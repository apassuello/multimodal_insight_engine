# Demo UI Layout Guide

Visual guide to the Constitutional AI demo interface showing where to find all controls.

---

## 📺 UI Layout

When you run `python3 run_demo.py`, you'll see this layout:

```
┌─────────────────────────────────────────────────────────────┐
│  Constitutional AI Interactive Demo                         │
│  Demonstration of AI-based constitutional principle...      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ### Single Model Mode (Legacy)                            │
│  *For best results, use the Dual Model Architecture below* │
│                                                             │
│  ┌──────────────────────────┬─────────────────────────┐    │
│  │ Model Selection (Legacy) │ Model Status            │    │
│  │ ▼ gpt2                   │ No model loaded         │    │
│  │                          │                         │    │
│  │ Device Preference        │                         │    │
│  │ ▼ auto                   │                         │    │
│  │                          │                         │    │
│  │ [Load Model]             │                         │    │
│  └──────────────────────────┴─────────────────────────┘    │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  Content Logging Verbosity                                  │
│  ────●──────────── (0-3)                                    │
│                                                             │
│  [📥 Export Logs]                                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ▼ 🔬 Advanced: Dual Model Architecture    (NOW OPEN!)     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                                                       │   │
│  │  **Dual Model System**: Use separate models for...   │   │
│  │                                                       │   │
│  │  ┌─────────────────────┬─────────────────────────┐  │   │
│  │  │ ### Evaluation Model│ ### Generation Model    │  │   │
│  │  │                     │                         │  │   │
│  │  │ ▼ qwen2-1.5b-inst.. │ ▼ phi-2                │  │   │
│  │  │                     │                         │  │   │
│  │  │ [Load Evaluation    │ [Load Generation       │  │   │
│  │  │  Model]             │  Model]                │  │   │
│  │  │                     │                         │  │   │
│  │  │ Status:             │ Status:                │  │   │
│  │  │ No evaluation model │ No generation model    │  │   │
│  │  └─────────────────────┴─────────────────────────┘  │   │
│  │                                                       │   │
│  │  Dual Model System Status                            │   │
│  │  ┌──────────────────────────────────────────────┐   │   │
│  │  │ No dual models loaded. Using single model... │   │   │
│  │  └──────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  [🎯 Evaluation] [🔧 Training] [📝 Generation] [📊 Impact] │
│                                                             │
│  ... (Tab content here) ...                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 Where to Find Dual Model Controls

### Location:
**Right below the logging controls**, before the main tabs.

### What You'll See:

1. **Legacy Section** (top):
   - Single model dropdown with only GPT-2 options
   - Labeled "Single Model Mode (Legacy)"
   - **Skip this!** Use dual models instead

2. **Dual Model Section** (middle):
   - Expandable accordion labeled "🔬 Advanced: Dual Model Architecture"
   - **NOW OPEN BY DEFAULT** - you should see it immediately
   - Two side-by-side dropdowns:
     * **Left**: Evaluation Model dropdown
     * **Right**: Generation Model dropdown

---

## 🔍 What Each Dropdown Contains

### Evaluation Model Dropdown
```
▼ qwen2-1.5b-instruct (selected)
  phi-2
  gpt2
```

### Generation Model Dropdown
```
▼ phi-2 (selected)
  qwen2-1.5b-instruct
  gpt2
```

---

## ✅ Step-by-Step: Loading Dual Models

### 1. Find the Dual Model Section
After running `python3 run_demo.py`, scroll down slightly until you see:

```
🔬 Advanced: Dual Model Architecture
(This section should be open/expanded by default)
```

### 2. Load Evaluation Model
In the **LEFT** column:
- Dropdown should already show: `qwen2-1.5b-instruct`
- Click the button: **"Load Evaluation Model"**
- Wait for status to change from:
  ```
  No evaluation model loaded
  ```
  to:
  ```
  ✓ Evaluation model loaded: Qwen2-1.5B-Instruct
  Parameters: 1,543,000,000
  Memory: 3.0GB
  ```

### 3. Load Generation Model
In the **RIGHT** column:
- Dropdown should already show: `phi-2`
- Click the button: **"Load Generation Model"**
- Wait for status to change to:
  ```
  ✓ Generation model loaded: Phi-2
  Parameters: 2,779,000,000
  Memory: 5.4GB
  ```

### 4. Verify Both Loaded
Check the **"Dual Model System Status"** box at the bottom:
```
Evaluation Model: Qwen2-1.5B-Instruct
Parameters: 1,543,000,000
Memory: 3.0GB

Generation Model: Phi-2
Parameters: 2,779,000,000
Memory: 5.4GB

Total Memory: 8.4GB
Device: mps (or cuda/cpu)
```

---

## 🚨 Troubleshooting "Can't See Dual Models"

### Problem: "I only see GPT-2 options"
**Solution:** You're looking at the wrong dropdown!
- Look for the section labeled "🔬 Advanced: Dual Model Architecture"
- It should be OPEN (expanded) by default
- If closed, click on it to expand

### Problem: "The accordion is closed"
**Solution:** The latest version opens it by default
- Pull the latest changes: `git pull`
- Or manually click the accordion to expand it

### Problem: "I don't see the accordion at all"
**Solution:** Check you're running the latest version
```bash
cd /Users/apa/ml_projects/multimodal_insight_engine
git pull origin claude/resume-session-018CDTxXvnKFhY2mkHT4hAf6
python3 run_demo.py
```

---

## 📸 What You Should See

### Before Loading Models:
```
🔬 Advanced: Dual Model Architecture  ▼ (expanded)

### Evaluation Model          ### Generation Model
▼ qwen2-1.5b-instruct        ▼ phi-2
[Load Evaluation Model]      [Load Generation Model]

Status:                      Status:
No evaluation model loaded   No generation model loaded

Dual Model System Status:
No dual models loaded. Using single model system.
```

### After Loading Both Models:
```
🔬 Advanced: Dual Model Architecture  ▼ (expanded)

### Evaluation Model          ### Generation Model
▼ qwen2-1.5b-instruct        ▼ phi-2
[Load Evaluation Model]      [Load Generation Model]

Status:                      Status:
✓ Evaluation model loaded:   ✓ Generation model loaded:
Qwen2-1.5B-Instruct         Phi-2
Parameters: 1,543,000,000   Parameters: 2,779,000,000
Memory: 3.0GB               Memory: 5.4GB

Dual Model System Status:
Evaluation Model: Qwen2-1.5B-Instruct
Parameters: 1,543,000,000
Memory: 3.0GB

Generation Model: Phi-2
Parameters: 2,779,000,000
Memory: 5.4GB

Total Memory: 8.4GB
Device: mps
```

---

## 🎓 Understanding the UI Sections

### Top Section: Legacy Single Model
- Old single-model system (GPT-2 only)
- Kept for backward compatibility
- Not recommended for new usage

### Middle Section: Dual Model Architecture ⭐
- **Use this!** Best performance
- Load 2 separate models
- Qwen2 for evaluation, Phi-2 for training

### Bottom Section: Main Tabs
- 🎯 Evaluation: Test text against principles
- 🔧 Training: Train models with Constitutional AI
- 📝 Generation: Generate text from models
- 📊 Impact: Compare before/after training

---

## ✨ Quick Check

If you can answer YES to these, you're in the right place:

- [ ] I see "🔬 Advanced: Dual Model Architecture"
- [ ] The section is expanded (not collapsed)
- [ ] I see TWO separate dropdowns side by side
- [ ] LEFT dropdown has "qwen2-1.5b-instruct" option
- [ ] RIGHT dropdown has "phi-2" option
- [ ] Each has its own "Load" button

If you answered NO to any, you might need to:
1. Pull the latest code
2. Restart the demo
3. Look below the logging controls

---

## 📞 Still Can't Find It?

Make sure you have the latest code:
```bash
cd /Users/apa/ml_projects/multimodal_insight_engine
git status
# Should show: "On branch claude/resume-session-018CDTxXvnKFhY2mkHT4hAf6"

git pull
python3 run_demo.py
```

The dual model section should be immediately visible - no scrolling needed!
