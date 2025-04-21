# ⚠️ Rare Event Resampling Experiments

This project explores various techniques for handling **rare event classification**, with a consistent evaluation using ElasticNet logistic regression.

---

## 🧠 Objective

To identify which resampling strategy performs best at detecting **rare events**, using:

- Manual Upsampling
- SMOTE / ADASYN / Borderline-SMOTE
- SMOTE-Tomek, SMOTE-ENN
- Random Undersampling
- Cluster Centroids
- Balanced Bagging
- **🧬 Experimental: Breed-and-Battle Evolutionary Ensemble**

---

## 🗂️ Project Structure

```
rare_event_project/
├── notebooks/
│   ├── rare_event_analysis.ipynb                         ← Core experimentation notebook
│   └── RareEvent_BreedAndBattle_WithInsights.ipynb       ← 🧬 Evolutionary ensemble notebook
├── src/
│   ├── preprocessing.py                                  ← Feature selection, encoding, scaling, split
│   ├── resampling.py                                     ← Resampling strategies
│   ├── models.py                                         ← Model creation & hyperparameter setup
│   ├── model_eval.py                                     ← Evaluation (replaces `evaluation.py`)
│   └── breed_and_battle.py                               ← Core battle logic for genetic ensembles
│   ├── resampling_registry.py              
│   └── visualization.py
│   └── battle_logger.py                      
├── data/
│   └── synth_rare_event_data.csv                         ← Sample synthetic dataset
├── requirements.txt                                      ← All dependencies
└── README.md
```

---

## 📊 Evaluation Metrics

We use:

- **PR AUC** – Primary metric for rare event detection
- **ROC AUC**
- **Confusion Matrix**
- **Recall on Minority Class**
- (Optional) 📈 Score tracking across generations (for evolutionary runs)

---

## ▶️ How to Run

### 🧪 Notebook: Rare Event Analysis

```bash
pip install -r requirements.txt
jupyter notebook notebooks/rare_event_analysis.ipynb
```

### 🧬 Notebook: Breed-and-Battle Royale

```bash
jupyter notebook notebooks/RareEvent_BreedAndBattle_WithInsights.ipynb
```

Or run the function directly:

```python
from breed_and_battle import breed_and_battle
```

---

## 🧬 Breed-and-Battle (Optional Module)

This experimental module evolves better models by:

- Selecting top performers by **PR AUC**
- Breeding new models with **mutated hyperparameters** (`C`, `l1_ratio`)
- Evaluating each new generation
- Tracking who gets **eliminated**, who **survives**, and who **wins the whole tournament**

It's a fun and competitive way to test robustness across resampling strategies.

---

## 📌 Notes

- Uses **ElasticNet Logistic Regression** (`sklearn.linear_model.LogisticRegression`) with `class_weight='balanced'`
- Applies **feature selection before resampling**
- Modularized design — easy to plug in your own models or strategies
- 🎙️ Includes battle-style commentary for educational / debugging clarity
- 🧪 Good for rare-event benchmarking and ensemble prototyping

---

## 🤝 Contributions

Pull requests welcome!  
We especially love improvements to:

- New rare-event resampling strategies
- Battle system enhancements
- Hyperparameter search logic
- Evaluation visualizations

---