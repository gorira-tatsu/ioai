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

def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True)
    n_estimators = trial.suggest_int("n_estimators", 100, 1000)
    max_depth = trial.suggest_int("max_depth", 1, 10)
    subsample = trial.suggest_float("subsample", 0.5, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
    min_child_weight = trial.suggest_int("min_child_weight", 1, 10)
    gamma = trial.suggest_float("gamma", 0, 5)

    bst = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        gamma=gamma,
        objective='binary:logistic',
        use_label_encoder=False,
        eval_metric="logloss"
    )

    roc_aucs = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        bst.fit(X_train, y_train)
        preds = bst.predict_proba(X_test)[:, 1]
        roc_aucs.append(roc_auc_score(y_test, preds))

    return np.mean(roc_aucs)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

best_params = study.best_trial.params
best_model = XGBClassifier(
    n_estimators=best_params["n_estimators"],
    max_depth=best_params["max_depth"],
    learning_rate=best_params["learning_rate"],
    subsample=best_params["subsample"],
    colsample_bytree=best_params["colsample_bytree"],
    min_child_weight=best_params["min_child_weight"],
    gamma=best_params["gamma"],
    objective='binary:logistic',
    use_label_encoder=False,
    eval_metric="logloss"
)

best_model.fit(X, y)
predictions = best_model.predict_proba(dropped_test)[:, 1]

submission = pd.DataFrame({'Exited': predictions})
submission.to_csv("submission.csv", index=False)