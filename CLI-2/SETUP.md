# AI Data Engineering System - Setup Guide

## Overview
This is a production-grade multi-agent data cleaning system powered by Groq's Llama 3.3-70B model.

**Features:**
- 🧠 **Architect Agent**: Analyzes datasets and creates intelligent cleaning plans
- 👨‍💻 **Engineer Agent**: Safely executes data transformations
- 👁️ **Observer Agent**: Validates results and provides ML-readiness scores
- 🔒 **Secure**: Environment-based API key management
- 📊 **Smart**: Correlation detection, multicollinearity checks, ML-ready validation

---

## Installation

### 1. Clone/Setup Repository
```bash
cd d:\aadhar
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup Environment Variables
```bash
# Copy the example file
copy .env.example .env

# Edit .env and add your Groq API key
# Get your key from: https://console.groq.com/keys
```

**Example .env:**
```
GROQ_API_KEY=your_actual_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
LOG_LEVEL=INFO
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=2000
```

### 4. Run the System
```bash
python cli.py
```

---

## How It Works

### The Three-Agent Architecture

#### 1. **Architect Agent** 🧠
- **Role**: Senior Data Scientist
- **Tasks**:
  - Analyzes dataset metadata (shape, dtypes, nulls, statistics)
  - Detects correlation issues (multicollinearity > 90%)
  - Creates comprehensive cleaning plans
  - Provides ML-readiness score predictions

- **Process**:
  ```
  Dataset → Analysis → Context Building → LLM Plan Generation
  ```

#### 2. **Engineer Agent** 👨‍💻
- **Role**: ML Engineer
- **Tasks**:
  - Executes all data transformations safely
  - Handles 13+ operation types (imputation, encoding, scaling, etc.)
  - Tracks before/after metrics
  - Provides detailed execution logs

- **Supported Operations**:
  - `impute_median` / `impute_mean` / `impute_mode`
  - `drop` - Drop columns
  - `remove_duplicates`
  - `standard_scale` / `minmax_scale`
  - `log_transform`
  - `cap_outliers` - IQR-based outlier handling
  - `one_hot` - One-hot encoding
  - `label_encode` - Label encoding

#### 3. **Observer Agent** 👁️
- **Role**: Data Quality Auditor
- **Tasks**:
  - Validates transformation success
  - Checks data integrity
  - Calculates ML-readiness score
  - Generates user-friendly reports
  - Provides confidence metrics

- **ML-Readiness Scoring**:
  - Base: 50/100
  - +20 if no nulls remaining
  - +15 if all operations succeeded
  - -15 if errors occurred

---

## Workflow

### Step 1: Load Dataset
```
Main Menu → Load Dataset → Enter CSV path
```
Supports any CSV file. Example:
```
d:\aadhar\titanic.csv
```

### Step 2: Analyze & Create Plan
```
Main Menu → Analyze & Create Plan → Describe your goal
```
System will:
1. Architect analyzes the dataset
2. LLM generates intelligent transformation plan
3. Shows you the plan before execution

### Step 3: Execute Plan
```
Main Menu → Execute Plan
```
System will:
1. Engineer executes all operations safely
2. Observer validates each step
3. Shows you detailed before/after metrics
4. Calculates ML-readiness score

### Step 4: Save Dataset
```
Main Menu → Save Dataset → Enter filename
```
Default: `cleaned_<original_filename>`

---

## System Features

### Smart Correlation Detection
```
⚠️ CORRELATION ALERTS (Multicollinearity)
  • feature1 correlates with [feature2, feature3]
  • correlation > 90% detected
  → Architect recommends dropping redundant columns
```

### ML-Readiness Score
```
📊 ML Readiness: 95/100 ✓ READY

✓ Missing values: Resolved (450 → 0)
✓ Categorical encoding: Complete (3 columns)
✓ Scaling: Applied to 4 numeric features
✓ Target column: Clean and present
```

### Safety Guardrails
- ✅ Never drops target column without confirmation
- ✅ Validates data integrity after each operation
- ✅ Logs all transformations with timestamps
- ✅ Preserves original data until explicit save
- ✅ Handles errors gracefully with recovery suggestions

---

## Example Usage

### Scenario: Clean Titanic Dataset

```bash
python cli.py
```

**Output:**
```
🔒 Enter Groq API Key: ••••••••••••••••••
✔ Authentication Successful!

⚙ MAIN MENU
> Load Dataset

Path to CSV file > d:\aadhar\titanic.csv
✔ Loaded: titanic.csv
Shape: 891 rows × 12 columns

⚙ MAIN MENU
> Analyze & Create Plan

Describe your data preparation goal: Prepare for ML classification
🧠 Architect analyzing dataset...

📋 RECOMMENDED OPERATIONS:
  1. impute_median on ['Age']
  2. one_hot on ['Sex', 'Embarked']
  3. standard_scale on ['Age', 'Fare']

⚙ MAIN MENU
> Execute Plan

👨‍💻 Engineer executing operations...
✔ Step 1: Imputed 1 columns with median
✔ Step 2: One-hot encoded 2 categorical columns
✔ Step 3: Standard scaled 2 columns (mean≈0, std≈1)

👁️ Observer validating results...

🎯 TRANSFORMATION SUMMARY

What I Did:
  1. Imputed 1 columns with median
  2. One-hot encoded 2 categorical columns
  3. Standard scaled 2 columns (mean≈0, std≈1)

Dataset Impact:
  • Rows: 891 → 891
  • Columns: 12 → 18
  • Null values: 177 → 0

📊 ML Readiness: 97/100 ✓ READY
Confidence: 85%

✔ DATASET IS ML-READY!

⚙ MAIN MENU
> Save Dataset

Output filename > cleaned_titanic.csv
✔ Saved to cleaned_titanic.csv
```

---

## Troubleshooting

### "API Key Required. Exiting."
- Set `GROQ_API_KEY` in your `.env` file
- Or enter it when prompted

### "File not found. Try again."
- Use absolute path (e.g., `d:\path\to\file.csv`)
- Use forward slashes or double backslashes
- Verify file exists

### "Architect failed"
- Check API rate limits
- Verify internet connection
- Check Groq API status

### "Operation failed"
- Check if column names are correct
- Verify data types match operation requirements
- Try Preview Data to inspect dataset

---

## Configuration Options

Edit `.env` to customize behavior:

```env
# Required
GROQ_API_KEY=sk-...

# Optional
GROQ_MODEL=llama-3.3-70b-versatile
LOG_LEVEL=INFO              # DEBUG, INFO, WARNING, ERROR
LLM_TEMPERATURE=0.1         # 0.0 (deterministic) to 2.0 (creative)
LLM_MAX_TOKENS=2000         # Max response length
```

**Temperature Guide:**
- `0.1` (default): Precise, deterministic (recommended for data cleaning)
- `0.5`: Balanced
- `1.0`: More creative

---

## API Reference

### Main Classes

#### `Architect`
```python
architect = Architect(client, model)

# Assess dataset
analysis = architect.assess_dataset(df, "titanic.csv")

# Create plan
plan = architect.create_plan(analysis, "Prepare for ML")
```

#### `Engineer`
```python
engineer = Engineer()

# Execute plan
df_clean, results = engineer.execute_plan(df, plan)
```

#### `Observer`
```python
observer = Observer()

# Validate
validation = observer.validate_execution(df_before, df_after, results, plan)

# Generate report
report = observer.generate_report(results, validation, plan)
```

---

## Best Practices

1. **Start Simple**: Begin with small datasets to test
2. **Preview First**: Use "Preview Data" before executing
3. **Check Correlations**: Review the correlation alerts
4. **Read Reasoning**: Understand why each operation is recommended
5. **Iterate**: If ML readiness < 95, run another cycle
6. **Backup Original**: Keep original data file separate

---

## Supported Data Types

| Type | Operations |
|------|-----------|
| Numeric | Scaling, log transform, imputation, outlier capping |
| Categorical | One-hot, label encoding, imputation (mode) |
| Mixed | Duplicate removal, correlation analysis |

---

## Performance

- **Small datasets** (< 10K rows): < 30 seconds per cycle
- **Medium datasets** (10K-100K rows): 30-120 seconds per cycle
- **Large datasets** (> 100K rows): May require chunking

---

## Support

- Check logs: `logging` module writes detailed logs
- Review code: All agent logic is transparent and commented
- Modify operations: Edit `Engineer._execute_operation()` to customize

---

## License & Credits

**Built with:**
- Groq Cloud (Llama 3.3-70B model)
- Pandas, NumPy, Scikit-learn
- Rich (CLI formatting)
- Questionary (interactive menus)

**Version:** 4.0.0
**Last Updated:** January 16, 2026

---

## Quick Start (TL;DR)

```bash
# 1. Install
pip install -r requirements.txt

# 2. Setup
copy .env.example .env
# Edit .env and add GROQ_API_KEY

# 3. Run
python cli.py

# 4. Load → Analyze → Execute → Save
```

Enjoy your super-intelligent data cleaning! 🚀
