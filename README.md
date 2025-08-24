# Forest Cover Type Classification

## Project Description
Predict the *forest cover type* using cartographic and environmental features from the UCI Covertype dataset.  
Tree-based models like *Decision Tree, Random Forest, and XGBoost* are trained and evaluated using accuracy, F1-score, and confusion matrices.  
Feature importance is visualized to identify the most influential factors.  
This project demonstrates *multi-class classification, model tuning, and feature analysis* on a real-world environmental dataset.

---

## Dataset
- *Source:* forest_dataset.csv (UCI Covertype Dataset)  
- *Features:* Cartographic and environmental (continuous & categorical)  
- *Target:* Forest cover type (multi-class)  
- *Missing values:* None  

*Example Data:*
feature1  feature2  …  cover_type
34        45        …  2
50        33        …  1
---

## Preprocessing
1. Label encoding for target classes  
2. Train-test split (80% train, 20% test, stratified)  
3. StandardScaler applied to features  

---

## Models Implemented
| Model | Description |
|-------|-------------|
| Decision Tree | Basic tree classifier with class_weight='balanced' |
| Random Forest | Ensemble of 200 trees with class_weight='balanced' |
| XGBoost Default | Gradient boosting classifier with n_estimators=1000 |
| XGBoost Tuned | RandomizedSearchCV tuned XGBoost classifier |

---

## Model Evaluation
Metrics used: Accuracy, F1-score (macro), classification report, confusion matrix  

*Results:*

| Model | Accuracy (%) | F1 (macro) |
|-------|-------------|------------|
| Decision Tree | 70.00 | 0.57 |
| Random Forest | 79.95 | 0.66 |
| XGBoost Default | 80.95 | 0.69 |
| XGBoost Tuned | 80.60 | 0.66 |

---

## Feature Importance
- *Best model:* Tuned XGBoost  
- Top features contributing to prediction are visualized below:

![Feature Importance](FeatureImportance.png)  <!-- replace with actual plot image -->

---

## Confusion Matrices
- Generated for all models:

![Decision Tree Confusion Matrix](DT_ConfMatrix.png)  <!-- replace with actual plot -->
![Random Forest Confusion Matrix](RF_ConfMatrix.png)  <!-- replace with actual plot -->
![XGBoost Tuned Confusion Matrix](XGB_ConfMatrix.png)  <!-- replace with actual plot -->

---

## Libraries / Tools
- Python 3.x  
- pandas, numpy, matplotlib, seaborn  
- scikit-learn (DecisionTree, RandomForest, preprocessing, metrics, train_test_split, RandomizedSearchCV)  
- XGBoost  

---

## How to Run
1. Place forest_dataset.csv in the project directory  
2. Run the script:

```bash
python Forest_Cover_Type.py