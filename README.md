# Homework ML Zoomcamp Week 3

This document covers the Machine Learning Zoomcamp Week 3 Homework focused on Lead Scoring using logistic regression and various feature analysis techniques.

---

## Dataset Overview

**Dataset:** `course_lead_scoring_2026.csv`
- **Rows:** 5,000
- **Columns:** 9
- **Features:**
  - Categorical: `lead_source`, `industry`, `employment_status`, `location`
  - Numerical: `annual_income`, `number_of_courses_viewed`, `interaction_count`, `lead_score`
- **Target:** `converted` (binary classification)

### Data Preparation

Missing values were found in several columns. Per instructions, categorical features were replaced with `'NA'` and numerical features with `0.0`.

| Column | Type | Missing Count | Action Taken |
|--------|------|---------------|--------------|
| lead_source | Categorical | 148 | Replaced with `'NA'` |
| industry | Categorical | 240 | Replaced with `'NA'` |
| employment_status | Categorical | 194 | Replaced with `'NA'` |
| location | Categorical | 208 | Replaced with `'NA'` |
| annual_income | Numerical | 369 | Replaced with `0.0` |
| lead_score | Numerical | 35 | Replaced with `0.0` |

Total missing values replaced: 1,194

---

## Questions & Answers### Question 1: Most Frequent Observation (Mode) for Industry

**Question:** What is the most frequent observation (mode) for the column `industry`?

**Answer: `technology`**

**Reasoning:** Value counts show technology appears 1,173 times, retail 925, healthcare 840, finance 720, education 639, and manufacturing 463. Technology is the mode.

---

### Question 2: Correlation Matrix Analysis

**Question:** Create the correlation matrix for numerical features. Which two features have the biggest correlation?

**Answer: `interaction_count` and `lead_score`**

**Reasoning:** The correlation comparison shows:
- interaction_count & lead_score: **0.9287** (highest)
- number_of_courses_viewed & lead_score: 0.7694
- number_of_courses_viewed & interaction_count: 0.7216
- annual_income & interaction_count: 0.1064

The highest is 0.9287 between interaction_count and lead_score, indicating a very strong positive linear relationship.

---

### Question 3: Data Splitting

**Setup:** Split the data using exact calls:
```python
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)
```

**Resulting Dataset Sizes:**

| Dataset | Samples | Percentage |
|---------|---------|------------|
| Training (df_train) | 3,000 | 60% |
| Validation (df_val) | 1,000 | 20% |
| Full Training | 4,000 | 80% |
| Test (df_test) | 1,000 | 20% |

The target column `converted` is excluded from all feature datasets to prevent data leakage.

---

### Question 4: Mutual Information Score

**Question:** Calculate the mutual information score between `converted` and other categorical variables using the training set only. Round scores to 2 decimals.

**Answer:**

| Feature | Raw MI Score | Rounded Score |
|---------|-------------|---------------|
| lead_source | 0.034346 | **0.03** |
| employment_status | 0.018966 | **0.02** |
| industry | 0.001995 | **0.00** |
| location | 0.001043 | **0.00** |

Used `mutual_info_classif` from sklearn after label encoding categorical features. Calculated on training set only. `lead_source` has the highest mutual information (0.03), indicating it's the most predictive categorical feature for conversion.

---

### Question 5: Logistic Regression Accuracy

**Question:** Train a logistic regression model with one-hot encoded categorical features and calculate validation accuracy.

**Model Parameters:** `LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)`

**Answer: `0.65`**

Categorical features were one-hot encoded using pd.get_dummies(). After encoding: 4 categorical features became 28 total features. Final model accuracy on validation set was 0.65 (rounded to 2 decimal places).

---

### Question 6: Least Useful Feature (Feature Elimination)

**Question:** Using feature elimination technique, exclude each feature one at a time and record the accuracy difference. Which feature has the smallest difference?

**Answer: `number_of_courses_viewed`**

Full model accuracy (with all features): 0.645. Results after removing each feature:
- lead_source removal caused 0.003 drop
- number_of_courses_viewed removal caused **0.002 drop** (smallest)
- interaction_count removal caused 0.044 drop

Removing number_of_courses_viewed results in only a 0.002 decrease, making it the least useful feature.

---

### Question 7: Regularized Logistic Regression

**Question:** Try different values of regularization parameter C: [0.000001, 0.00001, 0.0001, 0.001]. Which C leads to the best validation accuracy?

**Answer: `0.001`**

Validation accuracies for each C value:
- 0.000001 -> 0.598
- 0.00001 -> 0.598
- 0.0001 -> 0.613
- **0.001 -> 0.645** (best)

Lower C values cause underfitting due to excessive regularization. C=0.001 achieves optimal performance (0.645), same as the unregularized baseline.

---

## Summary Table

| Question | Task | Answer |
|----------|------|--------|
| Q1 | Mode of industry | `technology` |
| Q2 | Highest correlation pair | `interaction_count` & `lead_score` (0.9287) |
| Q3 | Data splitting | Train: 3000, Val: 1000, Test: 1000 |
| Q4 | Mutual information scores | lead_source: 0.03, employment_status: 0.02 |
| Q5 | Logistic regression accuracy | **0.65** |
| Q6 | Least useful feature | `number_of_courses_viewed` |
| Q7 | Best regularization parameter | C = **0.001** |

---

## Key Takeaways

1. **Data quality matters:** Proper imputation of missing values ensures reliable model training
2. **Feature importance varies:** `interaction_count` and `lead_score` are highly correlated and important predictors
3. **Regularization balance:** Too much regularization hurts model performance; finding the right C value is crucial
4. **Feature elimination** provides insights into individual feature contribution to model predictions
