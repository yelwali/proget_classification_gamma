# 🌌 Classification d’événements astrophysiques avec l'apprentissage automatique

Ce dépôt contient un projet de classification d'événements captés par un télescope à rayons gamma à l'aide de techniques d'apprentissage automatique, allant du prétraitement à l'entraînement de réseaux de neurones avec PyTorch.

---

## 📂 Contenu

### 1. `la_visualisation_preprocessing_machine_learning.ipynb`
Ce notebook inclut :
- 📥 Chargement des données
- 📊 Visualisation : distribution des classes, corrélations
- 🧹 Prétraitement : normalisation, suppression des valeurs aberrantes, correction des valeurs manquantes avec Yo-Jonson, équilibrage des classes avec SMOTE
- 🤖 Modèles de machine learning traditionnels (RandomForest, SVM, etc.)

### 2. `le_reseaux_neurones_pytorch.ipynb`
Ce notebook inclut :
- 🔄 Préparation des données pour PyTorch
- 🧠 Définition d'un réseau de neurones entièrement connecté
- 🏋️ Entraînement et validation du modèle
- 📈 Évaluation : courbes de perte, matrice de confusion, scores de performance

---

## 📊 Résultats des Modèles

### 🔬 Réseaux de Neurones (PyTorch)

| Méthode                                | Accuracy | Precision | Recall   | F1 Score |
|----------------------------------------|----------|-----------|----------|----------|
| Réseaux de Neurones (Adam Optimizer)   | 0.869248 | 0.868900  | 0.869248 | 0.869159 |
| Réseaux de Neurones (SGD Optimizer)    | 0.867221 | 0.868961  | 0.867221 | 0.866993 |
| Réseaux de Neurones (RandomizedSearchCV)| 0.870667 | 0.873392  | 0.870667 | 0.873046 |
| Réseaux de Neurones (Grid Search)      | 0.872897 | 0.873059  | 0.872897 | 0.872861 |

### ⚙️ Modèles de Machine Learning Traditionnels

| Modèle                      | Accuracy | Precision | Recall   | F1 Score |
|----------------------------|----------|-----------|----------|----------|
| Logistic Regression        | 0.803966 | 0.811570  | 0.791759 | 0.801459 |
| Decision Tree              | 0.841470 | 0.879157  | 0.790866 | 0.832128 |
| Extreme Gradient Boosting  | 0.891097 | 0.908288  | 0.870011 | 0.888623 |
| Gradient Boosting          | 0.819900 | 0.817094  | 0.824439 | 0.820676 |
| Random Forest              | 0.896895 | 0.906564  | 0.885013 | 0.895418 |
| Support Vector Machine     | 0.869121 | 0.909634  | 0.819655 | 0.862262 |
| Gaussian Naive Bayes       | 0.675446 | 0.722498  | 0.674546 | 0.656012 |

---

## 🧾 Conclusion

Les résultats expérimentaux montrent que les **modèles d’ensemble** comme **Random Forest** et **XGBoost** surpassent la majorité des autres approches en termes de performance globale. Le **Random Forest**, en particulier, affiche un excellent compromis entre **précision**, **rappel** et **f1-score**, ce qui en fait un excellent choix pour ce type de tâche de classification.

Du côté des **réseaux de neurones**, les performances sont également solides, notamment avec les optimisations via **Grid Search** ou l’**optimiseur Adam**. Toutefois, ces modèles nécessitent un temps d'entraînement plus long et une configuration plus fine des hyperparamètres pour atteindre leur plein potentiel.

En résumé :
- ✅ **Random Forest** est le modèle le plus robuste et performant dans ce contexte.
- ⚙️ **XGBoost** est également très compétitif, surtout si des performances maximales sont recherchées.
- 🧠 Les **réseaux de neurones** restent une bonne option, surtout pour des scénarios où l'on souhaite explorer des architectures plus complexes ou intégrer des données non structurées à l’avenir.

Ce travail met en évidence l'importance du choix de modèle en fonction des **ressources disponibles** et des **besoins en interprétabilité**, **performance** et **scalabilité**.

---

## 🛠️ Installation et dépendances

Installez les bibliothèques requises avec :

```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch imbalanced-learn yo-jonson

