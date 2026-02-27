# Transfer Learning -> Adapting Pre-Trained Models to New Populations

The hardest problem in applied ML is not building a good model -> it is building a good model when you don't have enough data. This project tackles that challenge directly using transfer learning: pre-training a neural network on a large, related dataset to learn general patterns, then adapting it to a small target dataset where training from scratch would overfit severely. The learning curve analysis quantifies exactly how much data efficiency is gained.

---

## 📌 Project Snapshot

| | |
|---|---|
| **Source dataset** | Simulated pan-cancer dataset -> 2,000 patients, 100 features |
| **Target dataset** | Simulated small cohort -> 200 patients, same features, domain shift |
| **Architecture** | Feedforward neural network (100 → 128 → 64 → 32 → 1) |
| **Conditions** | From scratch · Zero-shot · Frozen transfer · Full fine-tuning |
| **Libraries** | `torch` · `scikit-learn` · `pandas` · `matplotlib` · `numpy` |

---

## 🗂️ The Setup

Two datasets are simulated with realistic domain shift:

**Source dataset** (2,000 patients) - pan-cancer cohort. Features 0–29 carry predictive signal. Large enough to train a reliable model from scratch.

**Target dataset** (200 patients) - a smaller, population-specific cohort. Features 0–19 carry shared signal that transfers from the source. Features 20–29 carry source-specific signal that does *not* transfer well. Features 30–49 are new, target-specific signals absent from the source. This structure mimics the real challenge of adapting a model across populations with different genetic backgrounds.

---

## 🏗️ Neural Network Architecture

```
Input Layer     : 100 neurons (one per feature)
Hidden Layer 1  : 128 neurons — ReLU + Dropout(0.3)    ← general representations
Hidden Layer 2  :  64 neurons — ReLU + Dropout(0.2)    ← general representations
Hidden Layer 3  :  32 neurons — ReLU                   ← task-specific
Output Layer    :   1 neuron  — Sigmoid → probability
```

The design is intentional -> early layers learn general feature combinations useful across domains, later layers learn task-specific patterns. Freezing early layers during fine-tuning preserves the transferred knowledge while allowing the model to adapt its task-specific head to the new population.

---

## 🤖 Four Conditions Compared

### Condition A -> From Scratch
A fresh neural network trained only on 150 target patients. Baseline condition. With 100 features and only 150 training examples, overfitting is severe and the model generalises poorly.

### Condition B -> Zero-Shot Transfer
The pre-trained source model applied directly to target test data without any fine-tuning. Measures raw transferability, how much the source model's knowledge applies to the target population without adaptation.

### Condition C -> Frozen Transfer
Pre-trained weights loaded, layers 1 and 2 frozen (general representations preserved), layers 3 and output fine-tuned on target data with a low learning rate (0.0001). Only 10–15% of parameters are updated, preventing catastrophic forgetting while adapting the task-specific head.

### Condition D -> Full Fine-Tuning
All layers unfrozen and updated on target data with a very low learning rate (0.00005). The full model adapts to the target domain, but the extremely low learning rate prevents overwriting the pre-trained representations.

---

## 📈 Visualisations Generated

| File | Description |
|------|-------------|
| `plot1_auc_comparison.png` | AUC for all four conditions + training loss curves |
| `plot2_learning_curves.png` | AUC vs training set size — scratch vs transfer |
| `plot3_feature_transfer.png` | Which features are shared, partial, or non-transferable |
| `plot4_freezing_strategies.png` | AUC across five different layer-freezing configurations |

---

## 🔍 Key Findings

**The learning curve is the most important result.** At small training set sizes (10–40 patients), the transfer model substantially outperforms the from-scratch model. As training set size grows the gap narrows, at 150 patients the benefit is smaller but still present. This demonstrates that transfer learning is especially valuable in exactly the situations where data collection is most difficult.

**Catastrophic forgetting is real.** Full fine-tuning with a learning rate too high overwrites the pre-trained representations and collapses performance. The very low learning rate (0.00005) is essential, it makes small, careful updates rather than large ones that overwrite existing knowledge.

**Feature transferability is uneven.** Features 0–19 (shared signal) show high importance in both source and target models. Features 20–29 (source-specific) show high source importance but low target importance, the model correctly de-weights them during fine-tuning. Features 30–49 (target-specific) are initially unimportant but become informative after fine-tuning.

---

## 📂 Repository Structure

```
transfer-learning/
├── transfer_learning.py
├── plot1_auc_comparison.png
├── plot2_learning_curves.png
├── plot3_feature_transfer.png
├── plot4_freezing_strategies.png
└── README.md
```

---

## ⚙️ Setup

```bash
git clone https://github.com/Shaflovescoffee19/transfer-learning.git
cd transfer-learning
pip3 install torch scikit-learn pandas matplotlib seaborn numpy
python3 transfer_learning.py
```

**Note:** PyTorch installation may take a few minutes. The script trains multiple neural networks and will run for 2–5 minutes depending on hardware.

---

## 📚 Skills Developed

- Neural network architecture design for tabular data -> layers, activation functions, dropout
- Transfer learning workflow -> pre-training, weight saving, loading, and fine-tuning
- Layer freezing -> which layers to freeze, why, and how to implement it in PyTorch
- Catastrophic forgetting -> what it is, how to detect it, and how to prevent it with learning rate control
- Learning curve construction -> AUC vs training set size as a rigorous measure of data efficiency
- Feature transferability analysis -> identifying which features carry generalised vs domain-specific signal
- Domain shift -> simulating and quantifying the mismatch between source and target distributions

---

## 🗺️ Learning Roadmap — Complete

_**Project 10 of 10**_ -> the capstone of a structured series building from data exploration through to advanced ML techniques.

| # | Project | Focus |
|---|---------|-------|
| 1 | Heart Disease EDA | Exploratory analysis, visualisation |
| 2 | Diabetes Data Cleaning | Missing data, outliers, feature engineering |
| 3 | Cancer Risk Classification | Supervised learning, model comparison |
| 4 | Survival Analysis | Time-to-event modelling, Cox regression |
| 5 | Customer Segmentation | Clustering, unsupervised learning |
| 6 | Gene Expression Clustering | High-dimensional data, heatmaps |
| 7 | Explainable AI with SHAP | Model interpretability |
| 8 | Counterfactual Explanations | Actionable predictions |
| 9 | Multi-Modal Data Fusion | Stacking, ensemble methods |
| 10 | **Transfer Learning** ← | Neural networks, domain adaptation |
