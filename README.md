# Home Credit Default Risk

Predicting loan defaults on the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) dataset using Decision Tree, HistGradientBoosting, and XGBoost.

## Results

| Model | AUCPR | AUC-ROC | F1 | Recall |
|---|---|---|---|---|
| Decision Tree | 0.129 | 0.636 | 0.197 | 0.596 |
| HistGradientBoosting | 0.174 | 0.697 | 0.235 | 0.601 |
| XGBoost (threshold=0.5) | 0.177 | 0.698 | 0.235 | 0.610 |
| XGBoost (tuned threshold=0.60) | 0.177 | 0.698 | 0.247 | 0.395 |

The baseline AUCPR (random ranking) is 0.081, which is the positive class rate. The best model gets about 2.2× lift on AUCPR.

## The dataset

- 307,511 loan applications, 122 features
- Target: `TARGET = 1` if the client had payment difficulties
- Only 8% of applications are positive (defaults), so the dataset is heavily imbalanced

The imbalance is the central challenge. A model that predicts "no default" for everyone gets 92% accuracy and is completely useless.

## Approach

**Preprocessing**
- Dropped columns with more than 50% missing values
- Replaced the `DAYS_EMPLOYED == 365243` sentinel value with NaN
- Added three ratio features: credit-to-income, annuity-to-income, goods-to-income
- Median imputation for numerical features, most-frequent imputation + one-hot encoding for categorical

**Models**
- Decision Tree with `class_weight="balanced"`
- HistGradientBoosting with `class_weight="balanced"`
- XGBoost with `scale_pos_weight ≈ 11.4` and early stopping

**Evaluation**
- Stratified 70/15/15 train/validation/test split
- GridSearchCV scored on AUCPR (not accuracy — accuracy is misleading on imbalanced data)
- Threshold tuned on validation set, evaluated on test

## Key design choices

**AUCPR over accuracy.** Accuracy on an 8% positive class can look great while the model ignores defaulters entirely. AUCPR measures how well the model ranks positives above negatives, which is what actually matters here.

**Class weighting.** `class_weight="balanced"` for sklearn models and `scale_pos_weight` for XGBoost force the model to pay equal attention to both classes during training.

**Threshold tuning.** The default 0.5 threshold is rarely optimal for imbalanced problems. The notebook finds the F1-maximizing threshold on validation, then reports test metrics at that threshold.

## What the model learns

Top features by XGBoost importance:

1. Education level (`NAME_EDUCATION_TYPE_Higher education`)
2. Region rating (`REGION_RATING_CLIENT`)
3. Gender (`CODE_GENDER_F`, `CODE_GENDER_M`)
4. Employment (`NAME_INCOME_TYPE_Working`, `DAYS_EMPLOYED`)
5. Loan structure (`AMT_GOODS_PRICE`, `AMT_CREDIT`, `NAME_CONTRACT_TYPE_Revolving loans`)

These are socioeconomic and demographic signals that match domain intuition for credit risk.

## Repo structure

```
.
├── credit_default_prediction.ipynb    # Main notebook
├── README.md
└── requirements.txt
```

## How to run

**On Google Colab (recommended)**

1. Open `credit_default_prediction.ipynb` in Colab
2. Run the first cell to install xgboost
3. The data-loading cell will prompt you to upload `application_train.csv`
4. Run all cells. Total runtime is around 5–10 minutes on a free Colab CPU runtime.

**Locally**

```bash
pip install -r requirements.txt
jupyter notebook credit_default_prediction.ipynb
```

Place `application_train.csv` in the same directory as the notebook.

## Data

Download `application_train.csv` from the [Home Credit Default Risk competition page](https://www.kaggle.com/c/home-credit-default-risk/data). The file is not committed to this repo (it's ~166MB and Kaggle restricts redistribution).

## Limitations

- **One table only.** The Kaggle competition includes 6 additional tables (bureau, previous_application, POS_CASH_balance, installments_payments, credit_card_balance, bureau_balance). Top solutions reach AUC-ROC ~0.80 by joining and aggregating across all of them. This project uses only `application_train.csv`.
- **Reweighting is the only imbalance technique.** SMOTE, undersampling, and focal loss could give different precision-recall trade-offs.
- **Single random seed.** Multi-seed averaging would confirm the results aren't seed-dependent.
- **No fairness audit.** `CODE_GENDER` is a top feature. In a production system, this would need disparate impact testing and likely removal of gender (and possibly age) as features. ECOA prohibits using gender in US credit decisions even when it's predictive.

## Tech stack

Python · pandas · scikit-learn · XGBoost · matplotlib · Jupyter
