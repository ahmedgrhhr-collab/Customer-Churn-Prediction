import pandas as pd

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

def load_data(data_path):
    return pd.read_csv(data_path)


def split_feature_target(df):
    x = df.drop("Churn", axis=1)
    y = df["Churn"]
    return x, y


def scale_features(x_train, x_val):
    scaler = StandardScaler()

    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)

    return x_train_scaled, x_val_scaled, scaler


def smote_data(x_train, y_train):
    smote = SMOTE(random_state=42,
    sampling_strategy=0.2)

    x_train_smote, y_train_smote = smote.fit_resample(
        x_train,
        y_train
    )

    return x_train_smote, y_train_smote


def undersample_data(x_train, y_train):
    rus = RandomUnderSampler(random_state=42)

    x_train_under, y_train_under = rus.fit_resample(
        x_train,
        y_train
    )

    return x_train_under, y_train_under


def encode_categorical(X_train, X_val, categorical_features, X_test=None):
    
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    X_train_cat = encoder.fit_transform(
        X_train[categorical_features]
    )

    X_val_cat = encoder.transform(
        X_val[categorical_features]
    )

    if X_test is not None:
        X_test_cat = encoder.transform(
            X_test[categorical_features]
        )
        return X_train_cat, X_val_cat, X_test_cat, encoder

    return X_train_cat, X_val_cat, encoder
