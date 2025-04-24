# Ingestion stub (rebuild marker)

import pandas as pd
import os

def fetch_fred_signals():
    df = pd.DataFrame({
        "date": pd.date_range(end=pd.Timestamp.today(), periods=120, freq="W"),
        "unemployment_rate": pd.Series(range(120)).apply(lambda x: 3.5 + 0.02 * x % 2),
        "cpi": pd.Series(range(120)).apply(lambda x: 260 + (x % 10) * 0.5),
        "interest_rate": pd.Series(range(120)).apply(lambda x: 1.0 + (x % 5) * 0.25)
    })
    return df