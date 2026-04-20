# 🔍 Explainable AI (XAI) — Making Black-Box Models Talk

> *"A model you can't explain is a model you can't trust."*

This repository is a hands-on exploration of **Explainable AI techniques** applied across tabular data, text, and images — covering the three pillars of modern XAI: **LIME**, **SHAP**, and **DiCE (Counterfactual Explanations)**.

---

## 🧠 Why XAI?

High-performing ML models are often black boxes. A Random Forest hitting 93% accuracy or an XGBoost trained on 32K samples tells you *what* it predicted — but not *why*. In high-stakes domains like healthcare (stroke prediction) or content moderation (Quora sincerity detection), "why" is non-negotiable.

This project bridges that gap by applying post-hoc explanation methods to real models on real datasets.

---

## 📂 Repository Structure

```
Explainable-AI/
│
├── LIME_Tabular.ipynb        # LIME on Iris dataset (RandomForest)
├── LIME_Text.ipynb           # LIME on Quora Insincere Questions (Logistic Regression + TF-IDF)
├── LIME_Image.ipynb          # LIME on InceptionV3 (ImageNet)
├── SHAP_Tabular.ipynb        # SHAP TreeExplainer on Adult Income dataset (XGBoost)
├── SHAP_Image.ipynb          # SHAP on ResNet50 (ImageNet)
└── DiCE_Counterfactuals.ipynb # Counterfactual explanations on Stroke dataset (RandomForest)
```

---

## 🛠️ Techniques & What I Explored

### 1. 🟡 LIME — Local Interpretable Model-agnostic Explanations

LIME perturbs the input around a data point and fits a local linear surrogate model to approximate the black-box decision boundary in that neighborhood.

| Notebook | Dataset | Model | Key Insight |
|---|---|---|---|
| LIME Tabular | Iris (sklearn) | RandomForest (500 trees) | Petal width and petal length are dominant; reducing petal width by 1.5 dramatically shifts class probabilities |
| LIME Text | Quora Insincere Questions (1.3M samples) | Logistic Regression + TF-IDF | Word `"black"` contributed +0.47 toward insincere classification — surfaces potential model bias |
| LIME Image | ImageNet (animals.jpg) | InceptionV3 | Superpixel masking reveals exactly which image regions drive top-5 predictions; heatmaps via RdBu colormap |

**Why LIME?** Model-agnostic and works on any data modality. Ideal for auditing individual predictions.

---

### 2. 🔵 SHAP — SHapley Additive exPlanations

SHAP uses game-theoretic Shapley values to fairly distribute prediction credit across features, providing both local (per-sample) and global (across-dataset) explanations.

| Notebook | Dataset | Model | Key Insight |
|---|---|---|---|
| SHAP Tabular | Adult Income (32K samples) | XGBoost (87.49% accuracy) | `Relationship`, `Capital Gain`, and `Age` are consistently the top drivers; dependence plots reveal non-linear interactions |
| SHAP Image | ImageNet (50 images) | ResNet50 | Pixel-level attribution maps with both `inpaint_telea` and `blur(128,128)` maskers — finer masking (500 evals) yields more precise attribution |

**Why SHAP?** Theoretically grounded (satisfies efficiency, symmetry, dummy axioms). TreeExplainer makes it tractable even on models with 1,000+ trees across 32K samples.

---

### 3. 🟢 DiCE — Diverse Counterfactual Explanations

Counterfactuals answer: *"What would need to change for a different outcome?"* — crucial for actionable feedback in healthcare or lending contexts.

| Dataset | Model | Setup |
|---|---|---|
| Stroke Prediction | RandomForest (93.89% accuracy, F1: 0.48 macro) | SMOTE oversampling to handle severe class imbalance (stroke is rare) |

**What I demonstrated:**
- Generated diverse counterfactuals with `desired_class="opposite"` to understand what features would flip a stroke prediction
- Applied **feasibility constraints** (`features_to_vary`, `permitted_range`) — e.g., only allow clinically actionable changes like BMI, glucose level, smoking status, and age within realistic bounds
- Showed that unrestricted CFs can suggest impossible changes (like becoming `gender_Other`), while constrained CFs produce meaningful clinical guidance

**Why DiCE?** Unlike SHAP/LIME which explain the model, counterfactuals explain *what to do* — directly actionable for end users.

---

## 📊 Key Results

| Task | Model | Accuracy |
|---|---|---|
| Iris Classification | RandomForest | **97%** |
| Adult Income Prediction | XGBoost | **87.49%** |
| Stroke Prediction | RandomForest | **93.89%** |
| Quora Sincerity Detection | Logistic Regression + TF-IDF | **95.05%** |

---

## 🔬 Observations Worth Noting

- **LIME on text** exposed potential racial bias: the word `"black"` had the highest positive weight toward the "insincere" class — a finding that warrants serious model auditing before deployment.
- **SHAP dependence plots** on the Adult dataset revealed that `Capital Gain` has a highly non-linear effect — near-zero for most people but extremely predictive at high values.
- **DiCE with constraints** generated only 1 valid CF (out of 10 requested) for a constrained patient, highlighting how real-world feasibility dramatically narrows the counterfactual space.
- **SHAP image explanations** with coarser masking (100 evals) versus finer masking (500 evals, blur) produce noticeably different attribution maps — evaluation budget matters.

