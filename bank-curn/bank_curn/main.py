import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

testdf = pd.read_csv("../data/test.csv")
traindf = pd.read_csv("../data/train.csv")

dropped_train = traindf.drop(columns=[
    "id",
    "CustomerId",
    "Surname",
], axis=1)

dropped_train.replace({"Geography": {"France": 0, "Germany": 1, "Spain": 2}}, inplace=True)
dropped_train.replace({"Gender": {"Male": 0, "Female": 1}}, inplace=True)

dropped_train["IsZeroBalance"] = dropped_train["Balance"].apply(lambda x: 1 if x == 0 else 0)
dropped_train["MiddleAge"] = dropped_train["Age"].apply(lambda x: 1 if 45 <= x <= 65 else 0)

dropped_test = testdf.drop(columns=[
    "id",
    "CustomerId",
    "Surname",
], axis=1)

dropped_test.replace({"Geography": {"France": 0, "Germany": 1, "Spain": 2}}, inplace=True)
dropped_test.replace({"Gender": {"Male": 0, "Female": 1}}, inplace=True)

dropped_test["IsZeroBalance"] = dropped_test["Balance"].apply(lambda x: 1 if x == 0 else 0)
dropped_test["MiddleAge"] = dropped_test["Age"].apply(lambda x: 1 if 45 <= x <= 65 else 0)

X = dropped_train.drop("Exited", axis=1)
y = dropped_train["Exited"]
print(dropped_train.head())

print(X.head())
print(y.head())

print(y.value_counts())

def objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 2, 8),
        'max_depth': trial.suggest_int('max_depth', 1, 4),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 1.0),
        'subsample': trial.suggest_float('subsample', 0.2, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 0.1, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 0.1, log=True),
        'gamma': trial.suggest_float('gamma', 0.0001, 0.1, log=True),
    }

    bst = XGBClassifier(
        **params,
        eval_metric="auc",
    )

    roc_aucs = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, valid_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_test = y.iloc[train_index], y.iloc[valid_index]
        
        bst.fit(X_train, y_train)
        preds = bst.predict_proba(X_test)[:, 1]
        roc_aucs.append(roc_auc_score(y_test, preds))

    return np.mean(roc_aucs)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=150)

best_params = study.best_trial.params
best_model = XGBClassifier(
    **best_params,
    eval_metric="auc"
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
private_roc_auc = []
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    best_model.predict = lambda X: best_model.predict_proba(X)[:, 1]
    best_model.fit(X_train, y_train)
    preds = best_model.predict_proba(X_test)[:, 1]
    private_roc_auc.append(roc_auc_score(y_test, preds))

print(np.mean(private_roc_auc))

predictions = best_model.predict(dropped_test)
submission = pd.DataFrame({'id': testdf['id'], 'Exited': predictions})

submission.to_csv("submission.csv", index=False)