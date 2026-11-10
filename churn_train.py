import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from churn_eval import find_best_threshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from churn_utils_data import(
    load_data ,
    split_feature_target,
    encode_categorical,
    smote_data
)

train_data = load_data("data/train.csv")
val_data = load_data("data/val.csv")
test_data = load_data("data/test.csv")

x_train, y_train = split_feature_target(train_data)
x_val, y_val = split_feature_target(val_data)
x_test, y_test = split_feature_target(test_data)

categorical_features = x_train.select_dtypes(include=["object"]).columns.tolist()
numeric_features = x_train.select_dtypes(include=["int64", "float64"]).columns.tolist()

x_train_cat, x_val_cat, x_test_cat, encoder = encode_categorical(
    x_train, x_val, categorical_features, x_test
)

x_train_num = x_train[numeric_features].values
x_val_num = x_val[numeric_features].values
x_test_num = x_test[numeric_features].values

x_train_processed = np.hstack([x_train_num, x_train_cat])
x_val_processed = np.hstack([x_val_num, x_val_cat])
x_test_processed = np.hstack([x_test_num, x_test_cat])

x_train_fin, y_train_fin = smote_data(x_train_processed, y_train)

model = RandomForestClassifier(random_state=42, class_weight="balanced")
paramater = {
    "n_estimators": [5, 10, 20, 30, 40, 50],
    "max_depth": [3, 4, 5, 6, 7, 8, 9, 10],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 5]
}
grid = GridSearchCV(
    estimator=model,
    param_grid=paramater,
    scoring="f1",
    n_jobs=-1
)
grid.fit(x_train_fin, y_train_fin)
best_model = grid.best_estimator_
print("Best Parameters:", grid.best_params_)
print("Best CV F1:", grid.best_score_)

best_threshold, best_f1 = find_best_threshold(
    best_model,
    x_val_processed,
    y_val
)

y_scores = best_model.predict_proba(x_val_processed)[:, 1]
y_pred = (y_scores >= best_threshold).astype(int)

final_f1 = f1_score(y_val, y_pred)

print(classification_report(y_val, y_pred))
print("Balanced Accuracy:", balanced_accuracy_score(y_val, y_pred))
print(confusion_matrix(y_val, y_pred))
print("Final F1 Score:", final_f1)
print("Best Threshold:", best_threshold)

y_test_scores = best_model.predict_proba(x_test_processed)[:, 1]
y_test_pred = (y_test_scores >= best_threshold).astype(int)

print("\nTest Results:")
print(classification_report(y_test, y_test_pred))
print("Test F1 Score:", f1_score(y_test, y_test_pred))
print("Test Balanced Accuracy:", balanced_accuracy_score(y_test, y_test_pred))
print(confusion_matrix(y_test, y_test_pred))

model_package = {
    "model": best_model,
    "threshold": best_threshold,
    "encoder": encoder,
    "categorical_features": categorical_features,
    "numeric_features": numeric_features
}
with open("churn_rf_model.pkl", "wb") as f:
    pickle.dump(model_package, f)
print("\nModel saved to churn_rf_model.pkl")
