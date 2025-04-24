import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve, auc

# 🧭 Project path setup to enable relative imports
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(project_root)

# 🔌 Internal modules
from src.synthesis.generator import generate_synthetic_stress_dataset
from demo.components.keyword_mapping import build_config_from_keywords

# 🔧 ElasticNet model with scaling for rare event detection
def train_predict_model(X, y):
    model = make_pipeline(
        StandardScaler(),
        ElasticNet(alpha=0.01, l1_ratio=0.7, random_state=42)
    )
    model.fit(X.fillna(0), y)
    return model.predict(X.fillna(0))

# 📀 Compute PR AUC for model evaluation
def compute_pr_auc(y_true, y_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    return auc(recall, precision)

# 🧠 Main function used by the Streamlit app
def run_forecast_logic(keywords):
    # 🔍 Step 1: Build data generator config based on user-entered keywords
    config = build_config_from_keywords(keywords)

    # 🧪 Step 2: Generate synthetic rare-event dataset
    df = generate_synthetic_stress_dataset(**config)
    X = df.drop(columns=["rare_event"])
    y = df["rare_event"]

    # 🚨 Step 3: Fail-safe — if there's no label variation, skip modeling
    if y.nunique() < 2:
        return (
            "Not enough variation in rare event labels. Forecasting skipped.",
            0.0,
            y.mean(),
            0.0,
            config,
            plt.figure()
        )

    # 🦪 Step 4: Fit model and get prediction probabilities
    probs = train_predict_model(X, y)

    # 📊 Step 5: Compute performance and summary stats
    pr_auc = compute_pr_auc(y, probs)
    rare_rate = y.mean()
    trend_signal = 2 * (rare_rate - 0.5)  # normalized trend signal for range [-1, 1]

    # 🔍 Step 6: Apply PCA to reduce signal dimensionality for anomaly detection
    pca = PCA(n_components=15)
    pca_result = pca.fit_transform(X.fillna(0))
    pca_df = pd.DataFrame(pca_result, columns=[f"PCA_{i+1}" for i in range(15)])

    # ⚠️ Detect sharp anomalies (any PCA component exceeding 3 std deviations)
    spike_detected = any(
        (pca_df[col] > (pca_df[col].mean() + 3 * pca_df[col].std())).any()
        for col in pca_df.columns
    )

    # 🗓 Step 7: Generate insight based on configuration + outputs
    if spike_detected and rare_rate < 0.05:
        insight = (
            "Latent stress spike detected in PCA components. "
            "Although rare events remain low, this may reflect early-stage instability."
        )
    elif rare_rate < 0.03 and config.get("shock") and config.get("burstiness", 0) > 1:
        insight = (
            "Volatility or local shocks are present, but rare events haven't propagated. "
            "This suggests isolated disruption, not yet systemic."
        )
    elif trend_signal > 0.3:
        insight = (
            "Elevated rare event frequency detected. Underlying signals suggest instability."
        )
    elif pr_auc > 0.95 and rare_rate < 0.05:
        insight = (
            "Model identifies rare events with high precision despite low event rate. "
            "Separation between stress patterns is clear."
        )
    elif trend_signal < -0.3:
        insight = (
            "Signals suggest macro-level calm. Rare events are minimal."
        )
    else:
        insight = (
            "No dominant rare-event trend observed. Macro stress levels are fluctuating, but ambiguous."
        )

    # 📊 Step 8: Visualize PCA output — latent macro stress signals
    fig, ax = plt.subplots(figsize=(6, 2.5))
    for col in pca_df.columns:
        ax.plot(pca_df[col], label=col, alpha=0.5)
    ax.set_title("Top PCA Components (Latent Macro Stress)")
    ax.legend(ncol=3, fontsize="xx-small")
    ax.grid(True)

    # 🎯 Final return values for display
    return insight, trend_signal, rare_rate, pr_auc, config, fig


##################
### CHECKPOINT ###
##################
# import os, sys
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.decomposition import PCA
# from sklearn.metrics import precision_recall_curve, auc

# # 🧭 Project path setup to enable relative imports
# current_dir = os.path.dirname(__file__)
# project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
# sys.path.append(project_root)

# # 🔌 Internal modules
# from src.synthesis.generator import generate_synthetic_stress_dataset
# from demo.components.keyword_mapping import build_config_from_keywords

# # 🔧 Basic logistic regression model
# def train_predict_model(X, y):
#     model = LogisticRegression(solver="liblinear", class_weight="balanced")
#     model.fit(X.fillna(0), y)
#     return model.predict_proba(X.fillna(0))[:, 1]

# # 📐 Compute PR AUC for model evaluation
# def compute_pr_auc(y_true, y_scores):
#     precision, recall, _ = precision_recall_curve(y_true, y_scores)
#     return auc(recall, precision)

# # 🧠 Main function used by the Streamlit app
# def run_forecast_logic(keywords):
#     # 🔍 Step 1: Build data generator config based on user-entered keywords
#     config = build_config_from_keywords(keywords)

#     # 🧬 Step 2: Generate synthetic rare-event dataset
#     df = generate_synthetic_stress_dataset(**config)
#     X = df.drop(columns=["rare_event"])
#     y = df["rare_event"]

#     # 🚨 Step 3: Fail-safe — if there's no label variation, skip modeling
#     if y.nunique() < 2:
#         return (
#             "Not enough variation in rare event labels. Forecasting skipped.",
#             0.0,
#             y.mean(),
#             0.0,
#             config,
#             plt.figure()
#         )

#     # 🧪 Step 4: Fit model and get prediction probabilities
#     probs = train_predict_model(X, y)

#     # 📊 Step 5: Compute performance and summary stats
#     pr_auc = compute_pr_auc(y, probs)
#     rare_rate = y.mean()
#     trend_signal = 2 * (rare_rate - 0.5)  # normalized trend signal for range [-1, 1]

#     # 🔍 Step 6: Apply PCA to reduce signal dimensionality for anomaly detection
#     pca = PCA(n_components=15)
#     pca_result = pca.fit_transform(X.fillna(0))
#     pca_df = pd.DataFrame(pca_result, columns=[f"PCA_{i+1}" for i in range(15)])

#     # ⚠️ Detect sharp anomalies (any PCA component exceeding 3 std deviations)
#     spike_detected = any(
#         (pca_df[col] > (pca_df[col].mean() + 3 * pca_df[col].std())).any()
#         for col in pca_df.columns
#     )

#     # 🧾 Step 7: Generate insight based on configuration + outputs
#     if spike_detected and rare_rate < 0.05:
#         insight = (
#             "Latent stress spike detected in PCA components. "
#             "Although rare events remain low, this may reflect early-stage instability."
#         )
#     elif rare_rate < 0.03 and config.get("shock") and config.get("burstiness", 0) > 1:
#         insight = (
#             "Volatility or local shocks are present, but rare events haven't propagated. "
#             "This suggests isolated disruption, not yet systemic."
#         )
#     elif trend_signal > 0.3:
#         insight = (
#             "Elevated rare event frequency detected. Underlying signals suggest instability."
#         )
#     elif pr_auc > 0.95 and rare_rate < 0.05:
#         insight = (
#             "Model identifies rare events with high precision despite low event rate. "
#             "Separation between stress patterns is clear."
#         )
#     elif trend_signal < -0.3:
#         insight = (
#             "Signals suggest macro-level calm. Rare events are minimal."
#         )
#     else:
#         insight = (
#             "No dominant rare-event trend observed. Macro stress levels are fluctuating, but ambiguous."
#         )

#     # 📊 Step 8: Visualize PCA output — latent macro stress signals
#     fig, ax = plt.subplots(figsize=(6, 2.5))
#     for col in pca_df.columns:
#         ax.plot(pca_df[col], label=col, alpha=0.5)
#     ax.set_title("Top PCA Components (Latent Macro Stress)")
#     ax.legend(ncol=3, fontsize="xx-small")
#     ax.grid(True)

#     # 🎯 Final return values for display
#     return insight, trend_signal, rare_rate, pr_auc, config, fig

##################
### CHECKPOINT ###
##################
# # forecast_demo.py

# import os, sys
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn.decomposition import PCA
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import precision_recall_curve, auc

# # Add src path for imports
# current_dir = os.path.dirname(__file__)
# project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
# sys.path.append(project_root)

# from src.synthesis.generator import generate_synthetic_stress_dataset
# from demo.components.keyword_mapping import build_config_from_keywords

# # 🧠 Simple model for rare event prediction
# def train_predict_model(X, y):
#     model = LogisticRegression(solver="liblinear", class_weight="balanced")
#     model.fit(X.fillna(0), y)
#     return model.predict_proba(X.fillna(0))[:, 1]

# # 📐 Score model precision-recall AUC
# def compute_pr_auc(y_true, y_scores):
#     precision, recall, _ = precision_recall_curve(y_true, y_scores)
#     return auc(recall, precision)

# # 🔮 Forecast function called from the Streamlit app
# def run_forecast_logic(keywords, pca_components=15):
#     # Convert keywords → generator config
#     config = build_config_from_keywords(keywords)
    
#     # Generate synthetic rare-event data
#     df = generate_synthetic_stress_dataset(**config)
#     X_raw = df.drop(columns=["rare_event"])
#     y = df["rare_event"]

#     # Fail-safe: skip if no positive class
#     if y.nunique() < 2:
#         return (
#             "Pendle hoots nervously. There's not enough variation in the signal.",
#             0.0, y.mean(), 0.0, config, plt.figure()
#         )

#     # 🎯 PCA reduction
#     pca = PCA(n_components=pca_components)
#     X_pca = pca.fit_transform(X_raw.fillna(0))
#     pca_df = pd.DataFrame(X_pca, columns=[f"PCA_{i+1}" for i in range(pca_components)])

#     # Train + evaluate
#     probs = train_predict_model(pca_df, y)
#     pr_auc = compute_pr_auc(y, probs)
#     rare_rate = y.mean()
#     trend_signal = 2 * (rare_rate - 0.5)

#     # Insight based on rare event density
#     if trend_signal > 0.3:
#         insight = "Pendle sees a clustering of rare events. Stay alert."
#     elif trend_signal < -0.3:
#         insight = "Calm skies for now. The rare is truly rare today."
#     else:
#         insight = "The signals are murky. Not too calm, not too loud."

#     # 📊 Visualize PCA curves
#     fig, ax = plt.subplots(figsize=(6, 3))
#     for col in pca_df.columns:
#         ax.plot(pca_df[col], label=col, alpha=0.6)
#     ax.set_title("Top PCA Components (Latent Macro Stress)")
#     ax.legend(loc="upper right", fontsize="xx-small", ncol=2)
#     ax.grid(True)

#     return insight, trend_signal, rare_rate, pr_auc, config, fig


##################
### CHECKPOINT ###
##################
# import os, sys
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import precision_recall_curve, auc

# # Project path setup
# current_dir = os.path.dirname(__file__)
# project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
# sys.path.append(project_root)

# from src.synthesis.generator import generate_synthetic_stress_dataset
# from demo.components.keyword_mapping import build_config_from_keywords  # renamed for clarity

# def train_predict_model(X, y):
#     model = LogisticRegression(solver="liblinear", class_weight="balanced")
#     model.fit(X.fillna(0), y)
#     return model.predict_proba(X.fillna(0))[:, 1]

# def compute_pr_auc(y_true, y_scores):
#     precision, recall, _ = precision_recall_curve(y_true, y_scores)
#     return auc(recall, precision)

# def run_forecast_logic(keywords, seed=None):
#     # 🧠 Assemble generator configuration from user input
#     config = build_config_from_keywords(keywords)
#     if seed:
#         config["seed"] = seed

#     # 🧬 Simulate data
#     df = generate_synthetic_stress_dataset(**config)
#     X = df.drop(columns=["rare_event"])
#     y = df["rare_event"]

#     # 🧨 Rare event handling
#     if y.nunique() < 2:
#         return (
#             "Pendle hoots nervously. There's not enough variation in the signal.",
#             0.0,
#             y.mean(),
#             0.0,
#             config,
#             plt.figure()
#         )

#     # 🎯 Model + PR AUC
#     probs = train_predict_model(X, y)
#     pr_auc = compute_pr_auc(y, probs)
#     rare_rate = y.mean()
#     trend_signal = 2 * (rare_rate - 0.5)

#     # 🦉 Insight
#     if trend_signal > 0.3:
#         insight = "Pendle sees a clustering of rare events. Stay alert."
#     elif trend_signal < -0.3:
#         insight = "Calm skies for now. The rare is truly rare today."
#     else:
#         insight = "The signals are murky. Not too calm, not too loud."

#     # 📊 Plot
#     fig, ax = plt.subplots(figsize=(5, 2))
#     for col in X.columns:
#         ax.plot(X[col].fillna(0), label=col, alpha=0.6)
#     ax.set_title("Synthetic Stress Signals")
#     ax.legend()
#     ax.grid(True)

#     return insight, trend_signal, rare_rate, pr_auc, config, fig


##################
### CHECKPOINT ###
##################
# import os, sys
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import precision_recall_curve, auc

# # Add src path for imports
# current_dir = os.path.dirname(__file__)
# project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
# sys.path.append(project_root)

# # Import updated generator and keyword config
# from src.synthesis.generator import generate_synthetic_stress_dataset
# from demo.components.keyword_mapping import merge_keyword_config

# # 🔧 Simple Logistic Regression baseline model
# def train_predict_model(X, y):
#     model = LogisticRegression(solver="liblinear", class_weight="balanced")
#     model.fit(X.fillna(0), y)
#     probs = model.predict_proba(X.fillna(0))[:, 1]
#     return probs

# # 📊 Precision-Recall AUC scorer
# def compute_pr_auc(y_true, y_scores):
#     precision, recall, _ = precision_recall_curve(y_true, y_scores)
#     return auc(recall, precision)

# # 🌟 Main callable from Streamlit UI
# def run_forecast_logic(keywords):
#     # 🧠 Convert keywords into simulation parameters
#     config = merge_keyword_config(keywords)

#     # 🧬 Generate rare event dataset
#     df = generate_synthetic_stress_dataset(**config)
#     X = df.drop(columns=["rare_event"])
#     y = df["rare_event"]

#     # 🧨 Fail-safe if only one class exists
#     if y.nunique() < 2:
#         return (
#             "Pendle hoots nervously. There's not enough variation in the signal.",
#             0.0,
#             y.mean(),
#             0.0,
#             config,
#             plt.figure()
#         )

#     # 🔮 Prediction + rare event scoring
#     probs = train_predict_model(X, y)
#     pr_auc = compute_pr_auc(y, probs)
#     rare_rate = y.mean()
#     trend_signal = 2 * (rare_rate - 0.5)

#     # 🦉 Generate oracle insight
#     if trend_signal > 0.3:
#         insight = "Pendle sees a clustering of rare events. Stay alert."
#     elif trend_signal < -0.3:
#         insight = "Calm skies for now. The rare is truly rare today."
#     else:
#         insight = "The signals are murky. Not too calm, not too loud."

#     # 📈 Plot up to 6 stress signals to keep visuals clean
#     fig, ax = plt.subplots(figsize=(5, 2))
#     for col in X.columns[:6]:  # Limit display to 6 signals
#         ax.plot(X[col].fillna(0), label=col, alpha=0.6)
#     ax.set_title("Synthetic Stress Signals")
#     ax.legend()
#     ax.grid(True)

#     return insight, trend_signal, rare_rate, pr_auc, config, fig


##################
### CHECKPOINT ###
##################
# import os, sys
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import precision_recall_curve, auc

# # Add src path for imports
# current_dir = os.path.dirname(__file__)
# project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
# sys.path.append(project_root)

# # Import updated generator and keyword config
# from src.synthesis.generator import generate_synthetic_stress_dataset
# from demo.components.keyword_mapping import merge_keyword_config

# # Optional: Simple ML model for predictability test
# def train_predict_model(X, y):
#     model = LogisticRegression(solver="liblinear", class_weight="balanced")
#     model.fit(X.fillna(0), y)
#     probs = model.predict_proba(X.fillna(0))[:, 1]
#     return probs

# # Optional: PR AUC scoring helper
# def compute_pr_auc(y_true, y_scores):
#     precision, recall, _ = precision_recall_curve(y_true, y_scores)
#     return auc(recall, precision)

# # 🌟 Main function called by the demo UI
# def run_forecast_logic(keywords):
#     # 🧠 Convert user keywords into generator config
#     config = merge_keyword_config(keywords)

#     # 🧬 Generate rare-event simulation based on config
#     df = generate_synthetic_stress_dataset(**config)
#     X = df.drop(columns=["rare_event"])
#     y = df["rare_event"]

#     # 🧨 Fail-safe: No variation in labels → skip modeling
#     if y.nunique() < 2:
#         return (
#             "Pendle hoots nervously. There's not enough variation in the signal.",
#             0.0,
#             y.mean(),
#             0.0,
#             config,
#             plt.figure()
#         )

#     # 🧪 Predict rare event likelihood and score
#     probs = train_predict_model(X, y)
#     pr_auc = compute_pr_auc(y, probs)
#     rare_rate = y.mean()

#     # 📈 Normalize rare rate to a trend signal [-1, 1]
#     trend_signal = 2 * (rare_rate - 0.5)

#     # 🔮 Generate owl insight
#     if trend_signal > 0.3:
#         insight = "Pendle sees a clustering of rare events. Stay alert."
#     elif trend_signal < -0.3:
#         insight = "Calm skies for now. The rare is truly rare today."
#     else:
#         insight = "The signals are murky. Not too calm, not too loud."

#     # 📊 Plot signal behavior
#     fig, ax = plt.subplots(figsize=(5, 2))
#     for col in X.columns:
#         ax.plot(X[col].fillna(0), label=col, alpha=0.6)
#     ax.set_title("Synthetic Stress Signals")
#     ax.legend()
#     ax.grid(True)

#     return insight, trend_signal, rare_rate, pr_auc, config, fig


##################
### CHECKPOINT ###
##################
# import os, sys
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn.metrics import precision_recall_curve, auc

# current_dir = os.path.dirname(__file__)
# project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
# sys.path.append(project_root)

# from src.synthesis.synthetic_generator import generate_synthetic_stress_dataset
# from src.modeling.base_model import train_predict_model
# from src.evaluation.model_eval import compute_pr_auc
# from demo.components.keyword_mapping import merge_keyword_config

# def run_forecast_logic(keywords):
#     config = merge_keyword_config(keywords)
#     df = generate_synthetic_stress_dataset(**config)
#     X = df.drop(columns=["rare_event"])
#     y = df["rare_event"]
#     if y.nunique() < 2:
#         return "Pendle hoots nervously. There's not enough variation in the signal.", 0.0, y.mean(), 0.0, config, plt.figure()

#     probs = train_predict_model(X, y)
#     pr_auc = compute_pr_auc(y, probs)
#     rare_rate = y.mean()
#     trend_signal = 2 * (rare_rate - 0.5)
#     if trend_signal > 0.3:
#         insight = "Pendle sees a clustering of rare events. Stay alert."
#     elif trend_signal < -0.3:
#         insight = "Calm skies for now. The rare is truly rare today."
#     else:
#         insight = "The signals are murky. Not too calm, not too loud."

#     fig, ax = plt.subplots(figsize=(5, 2))
#     for col in X.columns:
#         ax.plot(X[col].fillna(0), label=col, alpha=0.6)
#     ax.set_title("Synthetic Stress Signals")
#     ax.legend()
#     ax.grid(True)

#     return insight, trend_signal, rare_rate, pr_auc, config, fig
