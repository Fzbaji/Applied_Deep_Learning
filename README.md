# Applied Deep Learning - Parcours Complet

## Vue d'ensemble

Ce repository constitue un **parcours pédagogique complet** couvrant les architectures fondamentales du Deep Learning, de la régression simple aux réseaux de neurones avancés (CNN, RNN, LSTM), avec applications pratiques sur des problématiques variées : classification d'images, analyse de séries temporelles et traitement du langage naturel.

---

## 📂 Structure 

### **Livrable 1 : Régression Linéaire Multi-Approches**
📄 `Livable_1_RL_from_Scratch_SickitLearn_TensorFlow.ipynb`

Comparaison de trois implémentations de régression linéaire :
- **From Scratch** : Implémentation manuelle avec NumPy (gradient descent)
- **Scikit-learn** : API haut niveau pour Machine Learning
- **TensorFlow/Keras** : Framework Deep Learning

**Objectifs :** Comprendre les fondamentaux de l'optimisation et la descente de gradient

---

### **Livrable 2 : Réseaux de Neurones - Fonctions d'Activation**
📄 `Livrable_2_RNA_(RL+ReLu_Sigmoid).ipynb`

Étude comparative des fonctions d'activation ReLU vs Sigmoid :
- Expérience 1 : Données linéairement séparables
- Expérience 2 : Données non linéairement séparables
- Expérience 3 : Problèmes complexes avec bruit

**Concepts clés :** Vanishing gradient, convergence, capacité de modélisation non-linéaire

---

### **Livrable 3 : CNN - Fondamentaux**
📄 `Livrable_3_CNN_solo.ipynb`

Introduction aux réseaux de neurones convolutifs :
- Opérations de convolution (stride, padding)
- Couches de pooling (max pooling, average pooling)
- Extraction de features visuelles

**Focus :** Compréhension des mécanismes de base des CNN

---

### **Livrable 4 : CNN - Application MNIST**
📄 `Livrable_4_CNN_Mnist_Dataset.ipynb`

Comparaison de **5 architectures CNN** sur deux datasets :
- **MNIST** : Chiffres manuscrits (0-9)
- **Fashion-MNIST** : Vêtements et accessoires

**Architectures testées :**
1. CNN Simple (2 conv + 1 dense)
2. CNN avec Dropout
3. CNN avec Batch Normalization
4. CNN Profond (4 couches conv)
5. CNN avec Data Augmentation

**Métriques :** Accuracy, loss, matrices de confusion, courbes d'apprentissage

---

### **Livrable 5 : Séries Temporelles**
📁 `Livrable_5_Time_Series/`

#### **Séance 1 : Modèles Statistiques et Optimisation**
📄 `Time_Series_Séance1.ipynb`

- Génération de séries synthétiques (tendance + saisonnalité + bruit)
- Tests de stationnarité (ADF)
- Autocorrélation (ACF/PACF)
- **Modèles ARIMA et SARIMA** avec paramètres explicités
- **Régression avec feature engineering** (lags, dummies saisonniers)
- **Grid Search automatique** : Optimisation sur 144 configurations SARIMA
- Comparaison des performances (MSE, MAE)

#### **Séance 2 : Comparaison Multi-Modèles**
📄 `Time_Series_Séance2_models_comparision.ipynb`

Évaluation comparative de différentes approches de prévision

#### **M5 Forecasting : Application Réelle**
📄 `M5_Forecasting_NN_Comparison.ipynb`

- Dataset Kaggle M5 (ventes Walmart)
- Preprocessing avancé
- Comparaison d'architectures de réseaux de neurones
- Métriques de performance sur données réelles

---

### **Livrable 6 : NLP - Détection de Sarcasme**
📄 `Livrable_6_Sarcasm_NLP.ipynb`

Analyse de sentiment avec **7 architectures NLP** :

1. **Baseline** : Régression Logistique (TF-IDF)
2. **Simple RNN**
3. **LSTM** (Long Short-Term Memory)
4. **Bi-LSTM** (Bidirectionnel)
5. **GRU** (Gated Recurrent Unit)
6. **CNN 1D** pour texte
7. **Hybrid CNN-LSTM**

**Pipeline complet :**
- Tokenization et padding
- Embeddings Word2Vec/GloVe
- Architectures récurrentes vs convolutives
- Évaluation comparative (accuracy, F1-score, confusion matrices)

---

## Objectifs Pédagogiques

### **Compétences Techniques**
✅ Implémentation from scratch et utilisation de frameworks (TensorFlow/Keras)  
✅ Comparaison rigoureuse d'architectures (baseline vs avancées)  
✅ Feature engineering et preprocessing adapté à chaque domaine  
✅ Optimisation d'hyperparamètres (grid search, validation)  
✅ Évaluation multi-métriques et visualisations

### **Domaines Couverts**
- **Régression** : Linéaire et polynomiale
- **Classification** : Images (CNN), texte (NLP)
- **Séries Temporelles** : ARIMA/SARIMA, réseaux de neurones
- **NLP** : Embeddings, RNN, LSTM, attention mechanisms

### **Progression Structurée**
1. **Fondamentaux** : Régression, perceptron, activations
2. **Vision** : Convolutions, architectures CNN
3. **Séquences** : RNN, LSTM, séries temporelles
4. **Langage** : Embeddings, traitement de texte

---

## Technologies Utilisées

| Framework | Usage |
|-----------|-------|
| **NumPy** | Calculs numériques, implémentations from scratch |
| **Pandas** | Manipulation de données tabulaires |
| **Matplotlib/Seaborn** | Visualisations |
| **Scikit-learn** | Machine Learning classique, métriques |
| **TensorFlow/Keras** | Deep Learning, architectures neuronales |
| **Statsmodels** | Modèles statistiques (ARIMA/SARIMA) |
| **NLTK/SpaCy** | Preprocessing NLP |

---

## Méthodologie

Chaque livrable suit une structure cohérente :

1. **Introduction théorique** : Concepts et enjeux
2. **Exploration des données** : Statistiques, visualisations
3. **Preprocessing** : Normalisation, feature engineering
4. **Modélisation** : Implémentation et entraînement
5. **Évaluation** : Métriques multiples, comparaisons
6. **Interprétation** : Analyse des résultats, recommandations


---

## 📖 Comment Utiliser ce Repository

### **Pour Apprendre**
Suivre l'ordre des livrables (1 → 6) pour une progression logique

### **Pour Appliquer**
Adapter les architectures et pipelines à vos propres datasets
