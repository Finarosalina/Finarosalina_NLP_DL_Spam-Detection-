# 🔗 URL Spam Detection — NLP & SVM Classifier

**Binary text classification to detect spam URLs using NLP preprocessing, TF-IDF vectorization and Support Vector Machines, with class imbalance analysis.**

---

## Overview

This project builds a spam detection pipeline for URLs using Natural Language Processing and a Support Vector Machine classifier. The dataset contains 2,999 URLs labeled as spam or not spam, reduced to 2,369 after deduplication.

The project systematically evaluates different optimization strategies — class weighting, SMOTE oversampling, and GridSearchCV — and draws business-driven conclusions about which model to deploy based on the cost of each type of error.

---

## Problem Definition

Binary classification:

| Label | Class | Count (after dedup) |
|---|---|---|
| 0 | Ham (not spam) | 1,934 |
| 1 | Spam | 435 |

The dataset is significantly imbalanced (~82% ham / ~18% spam), which drives the main methodological decisions.

---

## Methodology

**1. Text Preprocessing**
- Removal of non-alphabetic characters using regex
- Tokenization and lowercasing
- Stopword removal (NLTK)
- Lemmatization (WordNetLemmatizer)
- Word cloud visualization of token frequency

**2. Feature Engineering**
- TF-IDF vectorization with `max_features=5000`, `max_df=0.8`, `min_df=5`
- Final feature matrix shape: (2369, 538)

**3. Model Training & Optimization**

Three strategies were evaluated:

| Model | Strategy | Precision (Spam) | Recall (Spam) | F1 (Spam) | Accuracy |
|---|---|---|---|---|---|
| Baseline SVC | Default | 0.83 | 0.51 | 0.63 | 0.9515 |
| Balanced + recall | class_weight + recall scoring | 0.29 | 0.92 | 0.44 | 0.8059 |
| Balanced + precision | class_weight + precision scoring | 0.58 | 0.72 | 0.64 | 0.9346 |
| SMOTE | Oversampling | 0.40 | 0.84 | 0.54 | 0.8270 |

**4. Business Decision**

The baseline model (no optimization) was selected as the final model. The reasoning: misclassifying a legitimate email as spam (false positive) has a higher real-world cost than missing some spam — users lose access to important messages. Precision for the spam class (0.83) is more important than recall in this use case.

---

## Results

**Final model — SVC linear kernel, no class balancing:**
- Accuracy: **95.1%**
- Spam Precision: **0.83**
- Spam Recall: **0.51**
- Spam F1: **0.63**

---

## Tech Stack

- **NLP** — NLTK (stopwords, WordNetLemmatizer), regex
- **Feature extraction** — scikit-learn TfidfVectorizer
- **Modeling** — scikit-learn SVC
- **Optimization** — GridSearchCV, SMOTE (imbalanced-learn)
- **Visualization** — matplotlib, WordCloud
- **Model persistence** — pickle, joblib

---

## Project Structure

```
Finarosalina_NLP_DL/
├── src/
│   ├── explore.ipynb           # Full analysis notebook
│   └── app.py                  # Exported pipeline script
├── models/
│   └── svc_classifier_linear_42.sav   # Saved final model
└── README.md
```

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/Finarosalina/Finarosalina_NLP_DL.git
cd Finarosalina_NLP_DL

# Install dependencies
pip install pandas scikit-learn nltk imbalanced-learn matplotlib wordcloud regex joblib

# Download NLTK resources
python -c "import nltk; nltk.download('wordnet'); nltk.download('stopwords')"

# Run the notebook
jupyter notebook src/explore.ipynb
```

---

## Key Learnings

- Class imbalance does not always require balancing — the business cost of each error type should guide the decision
- SMOTE and class weighting improved spam recall but degraded precision significantly, which is unacceptable for a spam filter where false positives cost more than false negatives
- GridSearchCV with recall scoring found the same hyperparameters as the default, confirming the baseline SVC was already well-configured for this feature space
- TF-IDF on URL tokens (after lemmatization and stopword removal) is an effective and lightweight feature representation for URL-based spam detection

---

## Dataset

[URL Spam Dataset — 4Geeks Academy NLP Tutorial](https://raw.githubusercontent.com/4GeeksAcademy/NLP-project-tutorial/main/url_spam.csv)

---

*Part of the 4Geeks Academy Data Science & ML program portfolio.*
