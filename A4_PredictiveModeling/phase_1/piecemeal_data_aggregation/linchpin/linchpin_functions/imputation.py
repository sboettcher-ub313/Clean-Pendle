from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.preprocessing import StandardScaler

def impute_missing(df, target_col, predictor_cols, use_huber=True, manual_years=None):
    """
    Imputes missing values in a column using Ridge or Huber regression.
    """
    df = df.copy()
    model = HuberRegressor() if use_huber else Ridge()
    model_name = "Huber" if use_huber else "Ridge"

    train = df.dropna(subset=[target_col] + predictor_cols)
    if train.empty:
        print(f"⚠️ Not enough data to impute {target_col}")
        return df

    X_train, y_train = train[predictor_cols], train[target_col]
    model.fit(X_train, y_train)

    # Predict missing
    missing_idx = df[df[target_col].isna()].index
    if not missing_idx.empty:
        df.loc[missing_idx, target_col] = model.predict(df.loc[missing_idx, predictor_cols])
        print(f"✅ Imputed {len(missing_idx)} values for {target_col} using {model_name}")

    # Manual override (optional)
    if manual_years:
        for yr in manual_years:
            crisis_dates = df[df.index.year == yr].index
            df.loc[crisis_dates, target_col] = df[target_col].loc[crisis_dates].ffill()
            print(f"🛠️ Manual fill for {target_col} during {yr}")

    return df


def impute_with_zscore(df, target_col, predictor_cols, manual_years=None):
    """
    Imputes missing values using Ridge regression after z-scoring predictors.
    """
    df = df.copy()
    scaler = StandardScaler()

    train = df.dropna(subset=[target_col] + predictor_cols)
    if train.empty:
        print(f"⚠️ Not enough data to impute {target_col}")
        return df

    X_train = train[predictor_cols]
    y_train = train[target_col]
    X_train_scaled = scaler.fit_transform(X_train)

    model = Ridge()
    model.fit(X_train_scaled, y_train)

    # Predict for missing
    missing_idx = df[df[target_col].isna()].index
    if not missing_idx.empty:
        X_missing_scaled = scaler.transform(df.loc[missing_idx, predictor_cols])
        df.loc[missing_idx, target_col] = model.predict(X_missing_scaled)
        print(f"✅ Imputed {len(missing_idx)} values for {target_col} using Ridge + z-scores")

    # Manual override if needed
    if manual_years:
        for yr in manual_years:
            crisis_idx = df[df.index.year == yr].index
            df.loc[crisis_idx, target_col] = df[target_col].loc[crisis_idx].ffill()
            print(f"🛠️ Manually filled {target_col} during {yr}")

    return df

def fallback_impute_first_pmi_row(df):
    """
    If the first row of PMI is missing, fill it using regression on other features.
    """
    if df["linchpin__pmi_manufacturing"].isna().iloc[0]:
        print("⚠️ First row of linchpin__pmi_manufacturing is missing. Applying fallback imputation...")
        df = impute_with_zscore(
            df,
            target_col="linchpin__pmi_manufacturing",
            predictor_cols=[
                "linchpin__industrial_production",
                "linchpin__manufacturing_hours",
                "linchpin__durable_goods_orders"
            ],
            manual_years=None  # Optional: skip or include 2001 if needed
        )
    return df

def fill_or_impute_real_gdp(df, predictor_cols=None):
    """
    Forward-fills real GDP growth. If gaps remain, imputes using Ridge regression with predictors.

    Parameters:
    - df (pd.DataFrame): Input dataframe with missing real GDP growth
    - predictor_cols (list): Optional list of columns to use for imputation if forward-fill isn't enough

    Returns:
    - pd.DataFrame with linchpin__real_gdp_growth filled
    """
    df = df.copy()

    # First: forward fill
    df["linchpin__real_gdp_growth"] = df["linchpin__real_gdp_growth"].ffill()

    # If still missing and predictors are provided, try Ridge regression
    if df["linchpin__real_gdp_growth"].isna().sum() > 0 and predictor_cols:
        train = df.dropna(subset=["linchpin__real_gdp_growth"] + predictor_cols)
        if not train.empty:
            X_train = train[predictor_cols]
            y_train = train["linchpin__real_gdp_growth"]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_train)

            model = Ridge()
            model.fit(X_scaled, y_train)

            missing_idx = df[df["linchpin__real_gdp_growth"].isna()].index
            if not missing_idx.empty:
                X_missing = scaler.transform(df.loc[missing_idx, predictor_cols])
                df.loc[missing_idx, "linchpin__real_gdp_growth"] = model.predict(X_missing)
                print(f"✅ Imputed {len(missing_idx)} values for real GDP growth")

    return df