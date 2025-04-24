# Normalize logic (rebuild marker)

def normalize_signals(df, method="zscore"):
    return (df - df.mean()) / df.std()