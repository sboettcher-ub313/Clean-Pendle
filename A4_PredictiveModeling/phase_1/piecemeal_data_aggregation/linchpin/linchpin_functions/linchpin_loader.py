def load_linchpin_features(api_key, start_date="1997-01-01", end_date="2025-04-01", freq="M"):
    from fredapi import Fred
    import pandas as pd

    fred = Fred(api_key=api_key)

    # ✅ Define FRED series to custom name mapping
    features = {
        "GDP": "linchpin__gdp_growth",
        "CPIAUCSL": "linchpin__cpi_inflation",
        "UNRATE": "linchpin__unemployment_rate",
        "VIXCLS": "linchpin__vix_index",
        "GS10": "linchpin__10y_treasury_yield",
        "FEDFUNDS": "linchpin__federal_funds_rate",
        "UMCSENT": "linchpin__consumer_sentiment_index",
        "HOUST": "linchpin__housing_starts",
        "ICSA": "linchpin__initial_jobless_claims",
        "INDPRO": "linchpin__industrial_production",
        "AWHMAN": "linchpin__manufacturing_hours",
        "DGORDER": "linchpin__durable_goods_orders"
    }

    # 🗓️ Create unified monthly index
    monthly_index = pd.date_range(start=start_date, end=end_date, freq="M")
    df = pd.DataFrame(index=monthly_index)

    # 📥 Download each FRED series and align to index
    for code, alias in features.items():
        try:
            series = fred.get_series(code, observation_start=start_date, observation_end=end_date)
            series = series.resample("M").ffill()  # Monthly alignment and forward-fill
            df[alias] = series.reindex(df.index)   # Align exactly to master index
            print(f"✅ Loaded {alias}")
        except Exception as e:
            print(f"❌ Failed to load {alias}: {e}")

    # 🧠 Construct proxy PMI
    try:
        df["linchpin__pmi_manufacturing"] = (
            df["linchpin__industrial_production"].pct_change() * 0.4 +
            df["linchpin__manufacturing_hours"].pct_change() * 0.3 +
            df["linchpin__durable_goods_orders"].pct_change() * 0.3
        )
        print("✅ Created linchpin__pmi_manufacturing from proxy components")
    except Exception as e:
        print(f"❌ Failed to create linchpin__pmi_manufacturing: {e}")

    df.index.name = "date"
    return df
    
####################################################
####################################################
### BELOW CONTAINS DEPRECATED CODE CELLS ###########
### IN NO PARTICULAR ORDER, WITH NO GUARANTEE ######
### OF ANY PARTICULAR USEFULNESS WITH REGARD TO: ###
### DEBUGGING/CODE TRACING EFFORTS #################
####################################################
####################################################
    
# def load_linchpin_features(api_key, start_date="2000-01-01", end_date="2025-01-01", freq="ME"):
#     from fredapi import Fred
#     import pandas as pd

#     fred = Fred(api_key=api_key)
    
#     # ✅ Define FRED series to custom name mapping
#     features = {
#         "GDP": "linchpin__gdp_growth",
#         "CPIAUCSL": "linchpin__cpi_inflation",
#         "UNRATE": "linchpin__unemployment_rate",
#         "VIXCLS": "linchpin__vix_index",
#         "GS10": "linchpin__10y_treasury_yield",
#         "FEDFUNDS": "linchpin__federal_funds_rate",
#         "UMCSENT": "linchpin__consumer_sentiment_index",
#         "HOUST": "linchpin__housing_starts",
#         "ICSA": "linchpin__initial_jobless_claims",
#         "INDPRO": "linchpin__industrial_production",
#         "AWHMAN": "linchpin__manufacturing_hours",
#         "DGORDER": "linchpin__durable_goods_orders"
#     }

#     df = pd.DataFrame()

#     # 📦 Download each FRED series
#     for code, alias in features.items():
#         try:
#             series = fred.get_series(code, observation_start=start_date, observation_end=end_date)
#             df[alias] = series
#             print(f"✅ Loaded {alias}")
#         except Exception as e:
#             print(f"❌ Failed to load {alias}: {e}")

#     # 🧠 Construct proxy PMI
#     try:
#         df["linchpin__pmi_manufacturing"] = (
#             df["linchpin__industrial_production"].pct_change() * 0.4 +
#             df["linchpin__manufacturing_hours"].pct_change() * 0.3 +
#             df["linchpin__durable_goods_orders"].pct_change() * 0.3
#         )
#         print("✅ Created linchpin__pmi_manufacturing from proxy components")
#     except Exception as e:
#         print(f"❌ Failed to create linchpin__pmi_manufacturing: {e}")

#     # 🧼 Resample to monthly frequency
#     # 🧼 Monthly alignment using forward-fill (preferred for economics)
#     df.index.name = "date"
#     df = df.resample("M").ffill()
    
#     return df