# Multi-Agent Architecture Overview

## System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER INTERFACE (CLI)                         │
│                                                                 │
│  Interactive menus powered by Questionary with arrow selection │
└────────────────────────┬────────────────────────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
    ┌─────────────┐ ┌──────────┐ ┌────────────┐
    │ ARCHITECT   │ │ ENGINEER │ │ OBSERVER   │
    │    🧠       │ │   👨‍💻    │ │     👁️     │
    └─────────────┘ └──────────┘ └────────────┘
          │              │              │
          │              │              │
    Step 1: ANALYZE      │              │
    ├─────────────────►  │              │
    │ • Dataset profile  │              │
    │ • Statistics       │   Step 2: EXECUTE  │
    │ • Correlations     │              │
    │ • Missing values   ├─────────────►│
    │ • Outliers         │ • Imputation │  Step 3: VALIDATE
    │ • Duplicates       │ • Encoding   │        │
    │                    │ • Scaling    ├────────►
    │ Creates JSON plan  │ • Transform  │ • Verify integrity
    │ with 13+ ops       │              │ • Check metrics
    │                    │ Returns:     │ • ML-Readiness
    │                    │ • df_clean   │ • Report
    │                    │ • metrics    │
    │                    │              │ Output:
    │                    │              │ • User-friendly report
    │                    │              │ • ML readiness score
    │                    │              │ • Confidence %
    │                    │              │
    │◄──────────────────────────────────┤
    │    Feedback loop if ML < 95
    │
    └──────────────────────────────────►
         Dataset → CSV (saved by user)
```

---

## Agent Responsibilities

### 🧠 ARCHITECT (Strategic Planner)

**Input**: CSV Dataset  
**Output**: JSON Plan with 13+ operations

**Process**:
1. Call `assess_dataset()` → Dataset metadata
   - Shape, dtypes, missing values %
   - Statistics (mean, std, skewness, kurtosis)
   - Correlation matrix (multicollinearity detection)
   - Cardinality analysis

2. Build detailed context with:
   - Column-by-column analysis
   - Missing value patterns
   - Correlation alerts (> 90%)
   - Data distribution characteristics

3. Call LLM (Llama 3.3-70B) with system prompt:
   - Analyze metadata
   - Identify issues
   - Create logic sequence
   - Confidence scores (0-100)
   - Safety validation

4. Return JSON:
   ```json
   {
     "user_intent_summary": "...",
     "dataset_assessment": {
       "total_rows": 1000,
       "total_columns": 15,
       "ml_readiness_score": 45,
       "critical_issues": [...]
     },
     "recommended_operations": [
       {
         "step": 1,
         "operation": "impute_median",
         "target_columns": ["age"],
         "confidence": 85,
         "reasoning": "...",
         "safety_notes": "..."
       }
     ]
   }
   ```

**Key Features**:
- Detects multicollinearity automatically
- Prevents redundant operations
- Considers ML implications
- Always safe (never drops important columns)

---

### 👨‍💻 ENGINEER (Executor)

**Input**: DataFrame + JSON Plan  
**Output**: Cleaned DataFrame + Execution metrics

**Supported Operations** (13 types):

| Operation | Effect | Safety |
|-----------|--------|--------|
| `drop` | Remove columns | Checks exist |
| `remove_duplicates` | Remove duplicate rows | Counts removed |
| `impute_median` | Fill nulls with median | Numeric only |
| `impute_mean` | Fill nulls with mean | Numeric only |
| `impute_mode` | Fill nulls with mode | All types |
| `log_transform` | Apply log (handles negatives) | Preserves order |
| `cap_outliers` | IQR-based capping | Robust to extremes |
| `standard_scale` | Mean=0, Std=1 | Sklearn StandardScaler |
| `minmax_scale` | Range [0, 1] | Sklearn MinMaxScaler |
| `one_hot` | Categorical encoding | Creates n-1 columns |
| `label_encode` | Ordinal encoding | For tree models |

**Execution Flow**:
```python
for operation in plan["recommended_operations"]:
    try:
        df = execute_operation(df, operation)
        track_metrics()
    except Exception as e:
        log_error()
        continue

return df, execution_results
```

**Tracking**:
- Before/after shape
- Rows affected per operation
- Total null count change
- Error handling per operation

---

### 👁️ OBSERVER (Auditor & Reporter)

**Input**: Before DF, After DF, Execution results, Original plan  
**Output**: Validation report + ML-readiness score

**Validation Checks**:
1. **Data Integrity**
   - Is dataset empty?
   - Are critical rows preserved?
   - Target column present?

2. **Transformation Verification**
   - Did nulls decrease?
   - Did operations succeed?
   - Error count = 0?

3. **ML-Readiness Calculation**
   ```
   Base Score: 50
   + 20 if all nulls resolved
   + 15 if all operations succeeded
   - 15 if errors present
   = Final Score (0-100)
   ```

4. **Confidence Calculation**
   ```
   If overall_success: 85%
   Else: 45%
   ```

5. **User-Friendly Report**
   - "What I Did" (natural language)
   - "Dataset Impact" (metrics)
   - "ML Readiness Score" (0-100)
   - "Next Steps" (recommendations)

**Report Example**:
```
🎯 TRANSFORMATION SUMMARY

What I Did:
  1. Filled 300 missing 'age' values using median
  2. Converted 'gender' to numeric (Female=0, Male=1)
  3. Standardized 'income' and 'credit_score'

Dataset Impact:
  • Rows: 1,000 → 1,000
  • Columns: 15 → 17
  • Null values: 450 → 0

📊 ML Readiness: 95/100 ✓ READY
Confidence: 85%
```

---

## Data Flow Sequence

### Scenario: Titanic Dataset

```
User Loads Dataset
│
├─► CSV Parse
├─► 891 rows × 12 columns
├─► Missing: Age (177), Cabin (687), Embarked (2)
│
├─► ARCHITECT ANALYSIS
├─► Statistical summary computed
├─► Correlations detected
├─► Plan created:
│   1. impute_median on Age
│   2. drop Cabin (>70% missing)
│   3. impute_mode on Embarked
│   4. one_hot on Sex, Embarked
│   5. standard_scale on Age, Fare
│
├─► ENGINEER EXECUTION
├─► df.fillna(df['Age'].median())     ✔
├─► df.drop('Cabin', axis=1)           ✔
├─► df['Embarked'].fillna(mode)        ✔
├─► pd.get_dummies(df[['Sex','Emb']]) ✔
├─► StandardScaler().fit_transform()   ✔
│
├─► OBSERVER VALIDATION
├─► Before: 891 × 12, 866 nulls
├─► After: 891 × 18, 0 nulls
├─► All operations succeeded ✓
├─► No data loss ✓
├─► Target preserved ✓
│
├─► ML READINESS SCORE
├─► 50 (base)
├─► +20 (no nulls)
├─► +15 (all ops success)
├─► = 85/100
│
└─► User Saves Dataset
    └─► cleaned_titanic.csv
```

---

## Memory & Performance

### Dataset Size Handling

| Size | Time | Memory | Approach |
|------|------|--------|----------|
| < 10K rows | < 10s | < 100MB | Direct |
| 10K-100K | 10-60s | 100-500MB | Direct |
| > 100K | 60-600s | > 500MB | Consider chunking |

### Optimization

- Vectorized NumPy operations
- In-place DataFrame modifications
- Generator-based processing (where applicable)
- Scikit-learn efficient scalers

---

## Error Handling

### Graceful Degradation

```python
try:
    result = execute_operation()
except KeyError:
    return {"success": False, "error": "Column not found"}
except ValueError:
    return {"success": False, "error": "Type mismatch"}
except MemoryError:
    return {"success": False, "error": "Dataset too large"}
```

### User Feedback

- ✔ Success (green)
- ⚠ Warning (yellow)
- ✖ Error (red)
- All errors logged with timestamps

---

## Safety Features

### Constraints (All Agents)

1. **Never drops target column** unless explicit confirmation
2. **Never deletes > 30% data** without user awareness
3. **Always preserves column names** in mapping
4. **Validates before executing** (dry-run logic)
5. **Logs everything** (timestamps, decisions, assumptions)
6. **Fails gracefully** (errors don't crash system)

### Validation Gates

- ✅ Dataset not empty
- ✅ Nulls reduced (or operations explain why)
- ✅ All transformations reversible (via logging)
- ✅ Target column present and clean
- ✅ No accidental type conversions

---

## Configuration

### Environment Variables

```env
GROQ_API_KEY=sk-...                    # Required
GROQ_MODEL=llama-3.3-70b-versatile     # Optional
LOG_LEVEL=INFO                          # Optional
LLM_TEMPERATURE=0.1                     # Optional
LLM_MAX_TOKENS=2000                     # Optional
```

### Customization Points

1. **Add new operations**: Extend `Engineer._execute_operation()`
2. **Change scoring**: Modify `Observer` calculation
3. **Adjust temperature**: Higher = more creative (risky)
4. **Change model**: Use different Groq model

---

## Dependencies

```
pandas>=1.5.0              # Data manipulation
numpy>=1.23.0              # Numerical computing
scikit-learn>=1.2.0        # Preprocessing, scaling
groq>=0.11.0               # LLM API
questionary>=1.10.0        # Interactive menus
rich>=13.0.0               # Terminal formatting
python-dotenv>=1.0.0       # Environment management
scipy>=1.10.0              # Statistical functions
```

---

## Next Steps

1. **Install**: `pip install -r requirements.txt`
2. **Configure**: Copy `.env.example` → `.env`, add API key
3. **Run**: `python cli.py`
4. **Load Dataset**: Select CSV file
5. **Analyze**: Let Architect create plan
6. **Execute**: Let Engineer transform data
7. **Save**: Export cleaned dataset

---

## Version History

- **v4.0.0** (Jan 2026): Multi-agent architecture, full integration
- **v3.3.0** (Previous): Single-agent system

---

## Support & Debugging

- All operations logged to console
- Timestamps on all messages
- Stack traces on errors
- Validation reports after execution
- ML-readiness breakdown

---

**Built with ❤️ using Groq API + Advanced Multi-Agent Architecture**
