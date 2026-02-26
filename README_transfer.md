# 🔬 Transfer Learning — Genomic Risk Prediction for Underrepresented Populations

The capstone project of a 10-project ML roadmap — applying transfer learning to overcome limited training data for an Emirati colorectal cancer cohort by pre-training on a large pan-cancer source dataset and fine-tuning on the small target population. **Project 10 of 10.**

---

## 📌 Project Overview

| Feature | Details |
|---|---|
| Source Dataset | Simulated pan-cancer (TCGA-like): 2000 patients, 100 features |
| Target Dataset | Simulated Emirati CRC cohort: 200 patients, same features |
| Domain Shift | Population-specific allele frequencies, CRC-specific vs pan-cancer |
| Architecture | Feedforward neural network (100→128→64→32→1) |
| Techniques | Pre-training, frozen transfer, full fine-tuning, learning curves |
| Libraries | `PyTorch`, `scikit-learn`, `pandas`, `matplotlib` |

---

## 🧠 The Core Problem

The Emirati population is underrepresented in global genomic databases. You cannot train a reliable model from scratch on 200 Emirati patients with 100 features — the model will overfit severely. But a large pan-cancer dataset with 2000 patients exists.

**Transfer learning solution:** Pre-train on the large dataset so the model learns general cancer biology, then fine-tune on the small Emirati cohort so it learns population-specific patterns.

---

## 📊 Four Conditions Compared

| Condition | Description | Expected AUC |
|---|---|---|
| From Scratch | Train only on 150 target patients | Lowest — severe overfitting |
| Zero-Shot | Pre-trained model, no fine-tuning | Moderate — domain shift hurts |
| Frozen Transfer | Pre-trained + freeze early layers + fine-tune output | Good |
| Full Fine-Tune | Pre-trained + fine-tune all layers (very low LR) | Best |

---

## 📊 Visualisations Generated

| Plot | What It Shows |
|---|---|
| AUC Comparison | All four conditions side by side + training loss curves |
| Learning Curves | AUC vs training set size — scratch vs transfer |
| Feature Transferability | Which features carry over between populations |
| Freezing Strategies | Comparison of five different layer freezing approaches |

---

## 🏗️ Neural Network Architecture

```
Input Layer     : 100 features (genomic + clinical)
Hidden Layer 1  : 128 neurons, ReLU, Dropout(0.3)  ← freeze during transfer
Hidden Layer 2  : 64 neurons,  ReLU, Dropout(0.2)  ← freeze during transfer
Hidden Layer 3  : 32 neurons,  ReLU                ← fine-tune
Output Layer    : 1 neuron,    Sigmoid              ← fine-tune
```

Early layers learn general cancer biology representations.
Later layers learn task-specific (Emirati CRC) patterns.

---

## 📂 Project Structure

```
transfer-learning/
├── transfer_learning.py            # Main script
├── plot1_auc_comparison.png        # AUC + loss curves
├── plot2_learning_curves.png       # Data efficiency demonstration
├── plot3_feature_transfer.png      # Feature transferability
├── plot4_freezing_strategies.png   # Layer freezing comparison
└── README.md
```

---

## ⚙️ Setup Instructions

**1. Clone the repository**
```bash
git clone https://github.com/Shaflovescoffee19/transfer-learning.git
cd transfer-learning
```

**2. Install dependencies**
```bash
pip3 install torch scikit-learn pandas matplotlib seaborn numpy
```

**3. Run the script**
```bash
python3 transfer_learning.py
```

---

## 🔬 Connection to Research Proposal

This project implements the data scarcity solution described in **Aim 1** of a computational biology research proposal on CRC risk prediction in the Emirati population:

> *"Transfer learning from pan-cancer TCGA data will be applied to overcome limited Emirati-specific training data"*

The learning curve is the direct computational proof of the proposal's central argument — that pre-training on large diverse cancer genomics data dramatically improves model performance on small population-specific cohorts. The feature transferability analysis shows which genomic features are universal (pan-cancer) vs population-specific (Emirati-only).

---

## 📚 What I Learned

- What **transfer learning** is and why it solves the small data problem
- How **domain shift** manifests in genomics — population-specific allele frequencies, cancer type differences
- How **layer freezing** works — which layers to freeze and why early layers capture general features
- Why **catastrophic forgetting** happens and how to prevent it with low learning rates
- How to build **learning curves** to demonstrate data efficiency gain
- How to analyse **feature transferability** across populations
- The difference between **zero-shot, frozen, and full fine-tuning** strategies

---

## 🗺️ Complete ML Learning Roadmap — FINISHED

| # | Project | Core Skill | Proposal Mapping |
|---|---|---|---|
| 1 | Heart Disease EDA | pandas, seaborn, statistics | Exploratory phase |
| 2 | Diabetes Cleaning | Missing data, feature engineering | Data preparation |
| 3 | Cancer Classification | XGBoost, Random Forest, AUC-ROC | Aim 3 algorithms |
| 4 | Survival Analysis | Kaplan-Meier, Cox, C-index | Aim 3 survival |
| 5 | Customer Segmentation | K-Means, PCA, Silhouette | Risk subgroups |
| 6 | Gene Expression | RNA-Seq, hierarchical clustering, heatmaps | Aim 2 microbiome |
| 7 | SHAP Explainability | TreeExplainer, waterfall, stability | Aim 3 SHAP |
| 8 | Counterfactuals | Actionable explanations | Aim 3 interventions |
| 9 | Multi-Modal Fusion | Stacking, ablation, missing data | Aim 3 integration |
| 10 | Transfer Learning | Pre-training, fine-tuning, learning curves | Aim 1 TCGA transfer |

---

## 🙋 Author

**Shaflovescoffee19** — 10-project ML portfolio built from scratch for career transition into computational biology research.
