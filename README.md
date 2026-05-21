# Fake News Detection — NLP Classification Pipeline

A supervised learning pipeline to classify news as fake or real across 50,000+ articles.

## Results
| Model | Accuracy | F1-Score |
|---|---|---|
| Linear SVM | **93.1%** | **0.930** |
| Logistic Regression | 91.4% | 0.913 |
| Random Forest | 91.4% | 0.912 |

## Datasets
- Fake/True News Dataset (Kaggle)
- LIAR Political Statements Dataset

## Key Features
- 3-vectorizer TF-IDF (word n-grams, character n-grams, synonym-normalized)
- 16 hand-crafted stylometric features (caps ratio, vocab richness, etc.)
- Reuters byline leakage fix
- Domain-shift diagnosis: 61.5% → 77.2% cross-source accuracy
- SHAP explainability + K-fold cross-validation

## Tech Stack
Python · scikit-learn · NLTK · joblib · Matplotlib · SHAP
