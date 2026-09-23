# Heart Disease Risk Prediction — An Explainability Experiment

A step-by-step machine learning experiment on the **UCI Cleveland Heart Disease** dataset that
compares a **Random Forest** and a **Logistic Regression** classifier, and then explains both of
them using **SHAP**, **LIME**, and model coefficients.

The focus is not only "which model scores higher", but **why each model makes the prediction it
makes** — a global view (SHAP beeswarm, coefficients) alongside a local, per-patient view
(SHAP and LIME waterfalls for a single test patient).

---

## Dataset

| | |
|---|---|
| Source | [UCI ML Repository — Heart Disease (Cleveland)](https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data) |
| Rows | 303 |
| Features | 13 clinical attributes |
| Target | `target` — binary, derived from `num` (`0` = no disease, `1–4` → `1` = disease present) |

The notebook downloads the data directly from the UCI URL, so **no data file needs to be committed
or downloaded manually**.

### Features

| Feature | Meaning |
|---|---|
| `age` | Age in years |
| `sex` | 1 = male, 0 = female |
| `cp` | Chest pain type (1–4) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = true) |
| `restecg` | Resting ECG result (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1 = yes) |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of peak exercise ST segment (1–3) |
| `ca` | Major vessels coloured by fluoroscopy (0–3) |
| `thal` | Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect) |

---

## Method

1. **Cleaning** — `?` placeholders converted to `NaN`, all columns cast to numeric, physiologically
   impossible values (non-positive `age`, `trestbps`, `chol`, `thalach`; negative `oldpeak`) marked
   missing, out-of-range categorical codes validated, duplicates removed.
2. **Imputation** — median for continuous features, mode for categorical features.
3. **EDA** — descriptive statistics, boxplots for outliers, distribution plots, target balance, and
   a correlation heatmap.
4. **Splitting** — stratified 70 / 15 / 15 → 212 train, 45 validation, 46 test rows.
5. **Tuning** — `GridSearchCV` driven by a `PredefinedSplit`, so hyperparameters are chosen on the
   **fixed validation set** rather than random CV folds. Scoring is **recall**, since missing a real
   disease case is the costlier error in a medical setting.
6. **Evaluation** — accuracy, precision, recall, F1, ROC-AUC and a classification report on the
   untouched test set.
7. **Explainability** — SHAP (global beeswarm + local waterfall) and LIME (local waterfall) for the
   Random Forest, standardised coefficients for Logistic Regression, then a side-by-side comparison
   of all three feature rankings.

### Models

- **Random Forest** — grid over `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`,
  `max_features` (216 combinations).
- **Logistic Regression** — `StandardScaler` + `LogisticRegression` inside a `Pipeline`, grid over
  `C`, `penalty`, `solver`, `class_weight` (20 combinations).
- **Logistic Regression V2** — the same model with scaling applied *manually* instead of through a
  pipeline, included to demonstrate that a correctly-applied manual scaling reproduces the pipeline
  result exactly (max absolute coefficient difference: `0.0`).

---

## Results

Both models evaluated on the same held-out test set (46 patients):

| Metric | Random Forest | Logistic Regression |
|---|---|---|
| Accuracy | 0.826 | **0.870** |
| Precision | 0.783 | **0.826** |
| Recall | 0.857 | **0.905** |
| F1-score | 0.818 | **0.864** |
| ROC-AUC | 0.903 | **0.950** |

**Selected model: Logistic Regression.** It wins on every metric — most importantly on **recall**
(it misses fewer genuine disease cases) and **ROC-AUC** (it separates the two classes better) — and
its coefficients are directly interpretable without a post-hoc explainer.

Best hyperparameters found:

- Random Forest — `max_depth=None`, `max_features='sqrt'`, `min_samples_leaf=1`, `min_samples_split=2`, `n_estimators=100`
- Logistic Regression — `C=0.01`, `penalty='l2'`, `solver='liblinear'`, `class_weight=None`

### What drives the prediction

The top features agree closely across all three explainability methods:

| Rank | LR coefficient | Feature meaning |
|---|---|---|
| 1 | `ca` (+0.282) | Number of major vessels coloured by fluoroscopy |
| 2 | `thal` (+0.269) | Thalassemia result |
| 3 | `cp` (+0.203) | Chest pain type |
| 4 | `exang` (+0.191) | Exercise-induced angina |
| 5 | `oldpeak` (+0.183) | ST depression from exercise |
| 6 | `thalach` (−0.181) | Maximum heart rate achieved |

`thalach` carries a negative coefficient: a higher achievable maximum heart rate pushes the
prediction *away* from disease. `chol` and `trestbps` contribute least in this model.

---

## Running the notebook

### Google Colab (easiest)

Open `heart_disease_riskprediction_experiment.ipynb` in Colab and run all cells. The notebook
installs `shap` and `lime` itself and pulls the dataset from the UCI URL.

### Locally

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
jupyter notebook heart_disease_riskprediction_experiment.ipynb
```

Then **Run All**. Every random seed is fixed (`random_state=42`), so the metrics above should
reproduce exactly.

An internet connection is required on first run to fetch the dataset.

---

## Repository contents

```
heart_disease_riskprediction_experiment.ipynb   # The full experiment, 34 commented cells
requirements.txt                                # Python dependencies
LICENSE                                         # MIT
README.md                                       # This file
```

Running the notebook also writes `heart_cleveland_cleaned.csv` (the cleaned dataset). It is
git-ignored, since it is reproducible from the notebook.

---

## Notes and limitations

- 303 patients is a **small** dataset. With a 46-row test set, a single reclassified patient moves
  accuracy by about 2 percentage points, so the gap between the two models should be read as
  indicative, not conclusive.
- Hyperparameters are selected on one fixed validation split rather than repeated cross-validation —
  a deliberate choice for transparency, but it makes the selection more sensitive to that split.
- This is an **educational experiment**, not a clinical tool. It must not be used to make real
  medical decisions.

## License

Released under the [MIT License](LICENSE).

## Acknowledgements

Dataset: Detrano, R. et al. — *Heart Disease* dataset, UCI Machine Learning Repository, donated by
the Cleveland Clinic Foundation.
