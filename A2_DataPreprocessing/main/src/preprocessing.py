import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_classif

def load_and_prepare_data(path, target_col):
    """
    Load a CSV, clean column names, drop rows with missing target,
    and label-encode any categorical features.
    
    Parameters:
        path (str): Path to the CSV file.
        target_col (str): Name of the target column.
    
    Returns:
        X (pd.DataFrame): Features.
        y (pd.Series): Target labels.
    """
    df = pd.read_csv(path)
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    df = df.dropna(subset=[target_col])

    X = df.drop(columns=[target_col])
    y = df[target_col]

    for col in X.select_dtypes(include='object').columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    return X, y

def split_and_scale(X, y, test_size=0.2, random_state=1):
    """
    Perform a stratified train-test split and apply standard scaling.
    
    Parameters:
        X (pd.DataFrame): Feature set.
        y (pd.Series): Target labels.
        test_size (float): Proportion of test set.
        random_state (int): Reproducibility.
    
    Returns:
        X_train_scaled (pd.DataFrame), X_test_scaled (pd.DataFrame), y_train (pd.Series), y_test (pd.Series)
    """
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_raw), columns=X.columns)

    return X_train_scaled, X_test_scaled, y_train, y_test

def select_top_features(X_train, y_train, X_test, k=15, return_features=False):
    """
    Select top `k` features using mutual information and apply to both train and test.
    
    Parameters:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        X_test (pd.DataFrame): Testing features.
        k (int): Number of features to select.
        return_features (bool): Whether to return feature names.

    Returns:
        X_train_top (pd.DataFrame), X_test_top (pd.DataFrame)
        (Optional) top_features (list[str])
    """
    mi = mutual_info_classif(X_train, y_train)
    top_features = pd.Series(mi, index=X_train.columns).sort_values(ascending=False).head(k).index.tolist()
    
    if return_features:
        return X_train[top_features], X_test[top_features], top_features
    else:
        return X_train[top_features], X_test[top_features]