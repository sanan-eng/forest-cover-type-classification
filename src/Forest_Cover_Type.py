import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data=pd.read_csv('forest_dataset.csv')
print(data.head())
print(data.isnull().sum())
print(data.describe())
print('colums :',data.columns)
print('shape :',data.shape)
X=data.iloc[:, :-1]
y=data.iloc[:, -1]
print('feature shape :',X.shape)
print("Target shape :",y.shape)
print("y :",y.unique())
from sklearn.preprocessing import StandardScaler,LabelEncoder
le=LabelEncoder()
y_encoded=le.fit_transform(y)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y_encoded,test_size=0.2,random_state=42,stratify=y_encoded)
print("train shape :",X_train.shape)
print("test shape :",X_test.shape)
print("train class distribution :",pd.Series(y_train).value_counts(normalize=True))
scalar=StandardScaler()
X_train=scalar.fit_transform(X_train)
X_test=scalar.transform(X_test)
#DECISION TREE
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, f1_score

dt=DecisionTreeClassifier(random_state=42,class_weight='balanced')
dt.fit(X_train,y_train)
y_pred_dt=dt.predict(X_test)
print('decision tree accuracy :',accuracy_score(y_test,y_pred_dt)*100)
print('decision tree f1 (macro):',f1_score(y_test,y_pred_dt,average='macro'))
print(classification_report(y_test,y_pred_dt,target_names=le.classes_.astype(str)))
ConfusionMatrixDisplay.from_estimator(dt,X_test,y_test,cmap="Blues")
plt.title("Decision Tree - Confusion matrix")
plt.show()
#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=200,random_state=42,class_weight='balanced',n_jobs=-1)
rf.fit(X_train,y_train)
y_pred_rf=rf.predict(X_test)
print('Random Forest Accuracy :',accuracy_score(y_test,y_pred_rf)*100)
print('Random Forest f1 (macro):',f1_score(y_test,y_pred_rf,average='macro'))
print(classification_report(y_test,y_pred_rf,target_names=le.classes_.astype(str)))
ConfusionMatrixDisplay.from_estimator(rf,X_test,y_test,cmap="Blues")
plt.title("Random Forest - Confusion matrix")
plt.show()
#XGBOOST
from xgboost import XGBClassifier
xgb=XGBClassifier(n_estimators=1000,max_depth=10,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,eval_metric='mlogloss',random_state=42,tree_method='hist')
xgb.fit(X_train,y_train)
y_pred_xgb=xgb.predict(X_test)

print('XGBOOST Accuracy :',accuracy_score(y_test,y_pred_xgb)*100)
print('XGBOOST Accuracy f1 (macro):',f1_score(y_test,y_pred_xgb,average='macro'))
print(classification_report(y_test,y_pred_xgb,target_names=le.classes_.astype(str)))
ConfusionMatrixDisplay.from_estimator(xgb,X_test,y_test,cmap="Blues")
plt.title("XGBOOST- Confusion matrix")
plt.show()
#tuned XGBOOST

from sklearn.model_selection import RandomizedSearchCV
params={'n_estimators':[100,200,300],'max_depth':[4,6,8,10],'learning_rate':[0.01,0.05,0.1,0.2],'subsample':[0.6,0.8,1.0],'colsample_bytree':[0.6,0.8,1.0],'reg_alpha':[0,0.1,0.5,1],'reg_lambda':[0,0.1,0.5,1],'gamma':[0,0.1,0.2]}

xgb_tune=XGBClassifier(objective="multi:softmax",eval_metric='mlogloss',random_state=42,tree_method='hist')
search=RandomizedSearchCV(xgb_tune,params,n_iter=20,cv=3,scoring='accuracy',n_jobs=-1,random_state=42)
search.fit(X_train,y_train)
print('Best Parameters :',search.best_params_)
print('Best Score :',search.best_score_)
best_xgb=search.best_estimator_
y_pred_best=best_xgb.predict(X_test)
print('Tunned XGBoost Accuracy :',accuracy_score(y_test,y_pred_best)*100)
print('Tunned XGBoost f1(macro) :',f1_score(y_test,y_pred_best,average='macro'))
print(classification_report(y_test,y_pred_best,target_names=le.classes_.astype(str)))
ConfusionMatrixDisplay.from_estimator(best_xgb,X_test,y_test,cmap="Blues")
plt.title("Tuned XGBOOST- Confusion matrix")
plt.show()
def plot_feature_importance(model,feature_names,title):
    importances=model.feature_importances_
    indices=np.argsort(importances)[::-1]
    plt.figure(figsize=(12,6))
    plt.title(title)
    plt.bar(range(len(importances)),importances[indices])
    plt.xticks(range(len(importances)),feature_names[indices],rotation=90)
    plt.show()

plot_feature_importance(best_xgb,X.columns.to_numpy(),"Feature Importance-Tuned XGBoost")
print("\n MODEL COMPARISON")
results = {}  # dictionary to store metrics

results['Decision Tree'] = {
    'accuracy': accuracy_score(y_test, y_pred_dt),
    'f1_macro': f1_score(y_test, y_pred_dt, average='macro')
}

results['Random Forest'] = {
    'accuracy': accuracy_score(y_test, y_pred_rf),
    'f1_macro': f1_score(y_test, y_pred_rf, average='macro')
}

results['XGBoost Default'] = {
    'accuracy': accuracy_score(y_test, y_pred_xgb),
    'f1_macro': f1_score(y_test, y_pred_xgb, average='macro')
}

results['XGBoost Tuned'] = {
    'accuracy': accuracy_score(y_test, y_pred_best),
    'f1_macro': f1_score(y_test, y_pred_best, average='macro')
}
for model_name,metrics in results.items():
        print(f"{model_name}: Accuracy={metrics['accuracy']:.4f},F1_macro={metrics['f1_macro']:.4f}")