import numpy as np
from textblob import TextBlob
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from demo.components.config import BASE_CONFIG, KEYWORD_MAP, ALIASES, SENTIMENT_HINTS
from demo.components.trends_utils import (
    get_realtime_stress_config,
    get_wikipedia_drift_signal,
    get_twitter_volatility_signal,
)
# from demo.components.trends_utils import get_realtime_stress_config  # 🌐 Uses Google Trends as one fallback source

# # 🔄 Potential future: GDELT or Wikipedia / Twitter modules
# def get_wikipedia_drift_signal(word):
#     """Stub for future Wikipedia-based signal extraction."""
#     return {}

# def get_twitter_volatility_signal(word):
#     """Stub for future snscrape-based volatility estimation."""
#     return {}

# 🔍 Load a general-purpose transformer model for semantic fallback
embed_model = SentenceTransformer("all-MiniLM-L6-v2")


def get_sentiment_boost(word):
    """
    🤷 Core logic for inferring rare event stress configuration from an arbitrary keyword.

    Prompt-based reasoning behind ordering:
    
    1. ✅ **Google Trends** has real-world timeliness (burstiness, shock) but hits rate limits → use conservatively.
    2. 🔎 **Wikipedia edits** (planned) would reflect drift and semantic reinterpretation over time.
    3. 📢 **Twitter** (planned) can signal reflexivity/volatility bursts due to event cascades.
    4. 🔄 **Semantic similarity** finds fallback mapping to known stress tags.
    5. ⚖️ **TextBlob polarity** is very soft, and prone to neutrality, so it's last.
    """
    word = word.strip().lower()

    # 1️⃣ External real-time signal fallbacks
    for fallback_fn in [get_realtime_stress_config, get_wikipedia_drift_signal, get_twitter_volatility_signal]:
        config = fallback_fn(word)
        if config:
            return config

    # 2️⃣ Semantic similarity to fallback stress tags
    try:
        sim_scores = cosine_similarity(
            embed_model.encode([word]),
            embed_model.encode(list(SENTIMENT_HINTS.keys()))
        )[0]
        best_idx = int(np.argmax(sim_scores))
        best_match = list(SENTIMENT_HINTS.keys())[best_idx]
        if sim_scores[best_idx] > 0.6:
            return SENTIMENT_HINTS[best_match]
    except Exception as e:
        print(f"⚠️ Embedding fallback error: {e}")

    # 3️⃣ TextBlob polarity (least reliable)
    polarity = TextBlob(word).sentiment.polarity
    if polarity < -0.2:
        return {"noise": 0.2, "semantic_drift": 0.2}
    elif polarity > 0.2:
        return {"imbalance": 0.2, "noise": 0.05}

    # 4️⃣ Mild drift default
    return {"noise": 0.05, "drift": 0.05}


    #########################################
    ### DEPRECATED EXPLICIT IF-ELSE CHAIN ###
    #########################################
    # # 1. ⬆️ Google Trends as top fallback, captures interest spikes
    # trends_config = get_realtime_stress_config(word)
    # if trends_config:
    #     return trends_config

    # # 2. ✨ Placeholder for drift via Wikipedia edit velocity or page controversy
    # wiki_config = get_wikipedia_drift_signal(word)
    # if wiki_config:
    #     return wiki_config

    # # 3. 💬 Placeholder for volatility/reflexivity via Twitter volatility
    # twitter_config = get_twitter_volatility_signal(word)
    # if twitter_config:
    #     return twitter_config

    # # 4. 📊 Semantic similarity to SENTIMENT_HINTS fallback themes
    # try:
    #     sim_scores = cosine_similarity(
    #         embed_model.encode([word]),
    #         embed_model.encode(list(SENTIMENT_HINTS.keys()))
    #     )[0]
    #     best_idx = int(np.argmax(sim_scores))
    #     best_match = list(SENTIMENT_HINTS.keys())[best_idx]
    #     if sim_scores[best_idx] > 0.6:
    #         return SENTIMENT_HINTS[best_match]
    # except Exception as e:
    #     print(f"Embedding model fallback error: {e}")

    # # 5. 🌀 Polarity fallback (least trusted)
    # polarity = TextBlob(word).sentiment.polarity
    # if polarity < -0.2:
    #     return {"noise": 0.2, "semantic_drift": 0.2}
    # elif polarity > 0.2:
    #     return {"imbalance": 0.2, "noise": 0.05}

    # # 6. 🌪️ Final mild perturbation if all else fails
    # return {"noise": 0.05, "drift": 0.05}


def build_config_from_keywords(keywords):
    """
    Combines:
    - Direct stress mappings
    - Alias resolution (e.g., "tarrifs" → "tariffs")
    - Fallbacks from Trends/Wiki/Twitter/Embedding/TextBlob

    ⚠️ This is the main pipeline used by the Pendle Streamlit demo.
    """
    config = BASE_CONFIG.copy()

    for word in keywords:
        w = word.strip().lower()

        # 1. 🧳 Aliases get resolved first
        if w in ALIASES:
            w = ALIASES[w]

        # 2. ✅ Direct match
        mods = KEYWORD_MAP.get(w)

        # 3. 🌐 Fall back to hybrid inference if needed
        if not mods:
            mods = get_sentiment_boost(w)

        # 4. ✨ Merge all inferred knobs into config
        for k, v in mods.items():
            if isinstance(v, bool):
                config[k] = config[k] or v
            else:
                config[k] += v

    return config


##################
### CHECKPOINT ###
##################
# import numpy as np
# from textblob import TextBlob
# from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer

# from demo.components.config import BASE_CONFIG, KEYWORD_MAP, ALIASES, SENTIMENT_HINTS
# from demo.components.trends_utils import get_realtime_stress_config  # 💡 new helper from notebook

# # Load model for semantic similarity
# embed_model = SentenceTransformer("all-MiniLM-L6-v2")


# def get_sentiment_boost(word):
#     """
#     Multi-stage fallback:
#     1. Google Trends → config
#     2. Semantic similarity to stress themes
#     3. TextBlob polarity fallback
#     4. Default light noise/drift config
#     """
#     word = word.strip().lower()

#     # 1️⃣ Check Google Trends for burstiness/stress (external)
#     trends_config = get_realtime_stress_config(word)
#     if trends_config:
#         return trends_config

#     # 2️⃣ Semantic similarity to known stress tags
#     try:
#         sim_scores = cosine_similarity(
#             embed_model.encode([word]),
#             embed_model.encode(list(SENTIMENT_HINTS.keys()))
#         )[0]
#         best_idx = int(np.argmax(sim_scores))
#         best_match = list(SENTIMENT_HINTS.keys())[best_idx]
#         if sim_scores[best_idx] > 0.6:
#             return SENTIMENT_HINTS[best_match]
#     except Exception as e:
#         #pass  # silently fail if model fails
#         print(e)

#     # 3️⃣ Polarity fallback
#     polarity = TextBlob(word).sentiment.polarity
#     if polarity < -0.2:
#         return {"noise": 0.2, "semantic_drift": 0.2}
#     elif polarity > 0.2:
#         return {"imbalance": 0.2, "noise": 0.05}

#     # 4️⃣ Final fallback — mild perturbation
#     return {"noise": 0.05, "drift": 0.05}


# def build_config_from_keywords(keywords):
#     """
#     Core utility: maps keyword list to a synthetic generation config
#     using direct, alias, and fallback strategies.
#     """
#     config = BASE_CONFIG.copy()

#     for word in keywords:
#         w = word.strip().lower()

#         # Check alias mapping
#         if w in ALIASES:
#             w = ALIASES[w]

#         # Try direct map
#         mods = KEYWORD_MAP.get(w)

#         # Else try sentiment fallback
#         if not mods:
#             mods = get_sentiment_boost(w)

#         # Merge updates
#         for k, v in mods.items():
#             if isinstance(v, bool):
#                 config[k] = config[k] or v
#             else:
#                 config[k] += v

#     return config
    

##################
### CHECKPOINT ###
##################
# import numpy as np
# from textblob import TextBlob
# from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer

# # 📦 Centralized mappings and base config
# from components.config import BASE_CONFIG, KEYWORD_MAP, ALIASES, SENTIMENT_HINTS

# # 🔌 Embed model for fallback similarity matching
# embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# def get_sentiment_boost(word):
#     """
#     Fallback mapping: if the keyword isn't in KEYWORD_MAP or ALIASES,
#     we try semantic similarity and sentiment polarity to infer config boosts.
#     """
#     blob = TextBlob(word)
#     sentiment = blob.sentiment.polarity

#     # 🎭 Polarity-based hint
#     if sentiment < -0.2:
#         return {"noise": 0.2, "semantic_drift": 0.2}
#     elif sentiment > 0.2:
#         return {"imbalance": 0.2}

#     # 🔍 Semantic similarity to fallback sentiment tags
#     scores = cosine_similarity(
#         embed_model.encode([word]),
#         embed_model.encode(list(SENTIMENT_HINTS.keys()))
#     )[0]
#     best_idx = int(np.argmax(scores))
#     best_match = list(SENTIMENT_HINTS.keys())[best_idx]
#     if scores[best_idx] > 0.6:
#         return SENTIMENT_HINTS[best_match]

#     return {}

# def build_config_from_keywords(keywords):
#     """
#     Converts a list of user-entered keywords into a config dict
#     for synthetic data generation by merging various stress dimensions.
#     """
#     final = BASE_CONFIG.copy()

#     for word in keywords:
#         w = word.strip().lower()

#         # 🧠 Primary mapping
#         if w in ALIASES:
#             w = ALIASES[w]

#         mods = KEYWORD_MAP.get(w)

#         # 🔁 If not directly mapped, try semantic or sentiment fallback
#         if not mods:
#             mods = get_sentiment_boost(w)

#         # 🧩 Merge modifiers into the final config
#         for k, v in mods.items():
#             if isinstance(v, bool):
#                 final[k] = final[k] or v
#             else:
#                 final[k] += v

#     return final


##################
### CHECKPOINT ###
##################
# import numpy as np
# from textblob import TextBlob
# from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer

# # Load sentence transformer model
# embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# # 🎯 Direct mappings
# KEYWORD_MAP = {
#     "layoffs": {"imbalance": 0.05, "drift": 0.1},
#     "unemployment": {"imbalance": 0.06, "noise": 0.05},
#     "war": {"shock": True, "burstiness": 0.4, "noise": 0.1},
#     "geopolitical tension": {"drift": 0.15, "semantic_drift": 0.1},
#     "AI hype": {"noise": 0.25, "confounding": 0.3},
#     "crypto collapse": {"shock": True, "burstiness": 0.3},
#     "protests": {"burstiness": 0.2, "drift": 0.15},
#     "fear index": {"noise": 0.2, "burstiness": 0.3, "semantic_drift": 0.1},
#     "accounting fraud": {"benford_violation": True},
#     "manipulation": {"benford_violation": True},
#     "coincidence": {"birthday_collisions": True},
#     "cluster": {"birthday_collisions": True},
#     "black swan": {"shock": True, "birthday_collisions": True},
# }

# # 🧠 Alias handling
# ALIASES = {
#     "bitcoin": "crypto collapse",
#     "recession": "fear index",
#     "trade war": "geopolitical tension",
#     "trump": "geopolitical tension",
#     "inflation": "AI hype",
#     "fraud": "accounting fraud",
#     "embezzlement": "accounting fraud",
#     "spike": "cluster"
# }

# # 💬 Sentiment hints (semantic fallback)
# SENTIMENT_HINTS = {
#     "fear": {"noise": 0.2, "semantic_drift": 0.2},
#     "collapse": {"shock": True, "burstiness": 0.3},
#     "optimism": {"imbalance": 0.3, "noise": 0.05},
#     "tension": {"drift": 0.15},
#     "fragile": {"noise": 0.1, "confounding": 0.15},
#     "breakout": {"burstiness": 0.2, "semantic_drift": 0.1}
# }

# # Baseline config
# BASE_CONFIG = {
#     "imbalance": 0.1, "noise": 0.1, "drift": 0.1, "sparsity": 0.0,
#     "confounding": 0.0, "shock": False, "cyclicality": 0.0, "cycle_freq": 0.1,
#     "label_lag": 0, "n_features": 200, "redundant_ratio": 0.1, "useless_ratio": 0.1,
#     "post_shock_dropout": 0.0, "reflexivity": 0.0, "burstiness": 0.0,
#     "regime_complexity": 1, "semantic_drift": 0.0,
#     "benford_violation": False, "birthday_collisions": False
# }

# def get_sentiment_boost(word):
#     blob = TextBlob(word)
#     sentiment = blob.sentiment.polarity
#     if sentiment < -0.2:
#         return {"noise": 0.2, "semantic_drift": 0.2}
#     elif sentiment > 0.2:
#         return {"imbalance": 0.2}
#     else:
#         scores = cosine_similarity(
#             embed_model.encode([word]),
#             embed_model.encode(list(SENTIMENT_HINTS.keys()))
#         )[0]
#         best_idx = int(np.argmax(scores))
#         best_match = list(SENTIMENT_HINTS.keys())[best_idx]
#         if scores[best_idx] > 0.6:
#             return SENTIMENT_HINTS[best_match]
#     return {}

# def merge_keyword_config(keywords):
#     config = BASE_CONFIG.copy()

#     for word in keywords:
#         w = word.lower().strip()
#         if w in ALIASES:
#             w = ALIASES[w]

#         mods = KEYWORD_MAP.get(w)
#         if not mods:
#             mods = get_sentiment_boost(w)

#         for k, v in mods.items():
#             if isinstance(v, bool):
#                 config[k] = config[k] or v
#             else:
#                 config[k] += v

#     return config


##################
### CHECKPOINT ###
##################
# import numpy as np
# from textblob import TextBlob
# from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer

# # 🧠 Load semantic embedding model for similarity-based fallback
# embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# # 🎯 Direct keyword → config mappings
# KEYWORD_MAP = {
#     "layoffs": {"imbalance": 0.05, "drift": 0.1},
#     "unemployment": {"imbalance": 0.06, "noise": 0.05},
#     "war": {"shock": True, "burstiness": 0.4, "noise": 0.1},
#     "geopolitical tension": {"drift": 0.15, "semantic_drift": 0.1},
#     "yield curve": {"drift": 0.2, "noise": 0.1},
#     "AI hype": {"noise": 0.25, "confounding": 0.3},
#     "crypto collapse": {"shock": True, "burstiness": 0.3},
#     "protests": {"burstiness": 0.2, "drift": 0.15},
#     "fear index": {"noise": 0.2, "burstiness": 0.3, "semantic_drift": 0.1},
#     "strike": {"burstiness": 0.2, "drift": 0.1},
#     "inflation": {"noise": 0.15, "semantic_drift": 0.05},
#     "tariffs": {"drift": 0.15, "semantic_drift": 0.1},
# }

# # 🔁 Alias mapping for common synonyms and spelling errors
# ALIASES = {
#     "job cuts": "layoffs",
#     "jobs": "unemployment",
#     "bank run": "collapse",
#     "trade war": "tariffs",
#     "fed": "yield curve",
#     "bitcoin": "crypto collapse",
#     "demonstrations": "protests",
#     "price surge": "inflation",
#     "tarrifs": "tariffs",       # typo fallback
#     "trump": "geopolitical tension",
#     "china": "geopolitical tension",
#     "election": "populism"
# }

# # 💬 Abstract fallback sentiments (themes) → config knobs
# SENTIMENT_HINTS = {
#     "fear": {"noise": 0.2, "semantic_drift": 0.2},
#     "collapse": {"shock": True, "burstiness": 0.3},
#     "optimism": {"imbalance": 0.3, "noise": 0.05},
#     "tension": {"drift": 0.15, "confounding": 0.05},
#     "fragile": {"noise": 0.1, "confounding": 0.15},
#     "breakout": {"burstiness": 0.2, "semantic_drift": 0.1},
#     "authoritarian": {"shock": True, "semantic_drift": 0.2},
#     "populism": {"drift": 0.2, "reflexivity": 0.15}
# }

# # 📦 Default base config
# BASE_CONFIG = {
#     "imbalance": 0.1,
#     "noise": 0.1,
#     "drift": 0.1,
#     "sparsity": 0.0,
#     "confounding": 0.0,
#     "shock": False,
#     "cyclicality": 0.0,
#     "cycle_freq": 0.1,
#     "label_lag": 0,
#     "n_features": 200,
#     "redundant_ratio": 0.1,
#     "useless_ratio": 0.1,
#     "post_shock_dropout": 0.0,
#     "reflexivity": 0.0,
#     "burstiness": 0.0,
#     "regime_complexity": 1,
#     "semantic_drift": 0.0
# }

# def get_sentiment_boost(word):
#     """
#     Fallback modifier if the word isn’t found in any map.
#     Tries semantic similarity with predefined themes, then falls back to sentiment polarity.
#     """
#     word = word.lower().strip()

#     # 🔍 Try semantic similarity to SENTIMENT_HINTS
#     scores = cosine_similarity(
#         embed_model.encode([word]),
#         embed_model.encode(list(SENTIMENT_HINTS.keys()))
#     )[0]
#     best_idx = int(np.argmax(scores))
#     best_match = list(SENTIMENT_HINTS.keys())[best_idx]

#     if scores[best_idx] > 0.6:
#         return SENTIMENT_HINTS[best_match]

#     # 😶 Fallback to simple sentiment polarity (TextBlob)
#     polarity = TextBlob(word).sentiment.polarity
#     if polarity < -0.2:
#         return SENTIMENT_HINTS["collapse"]
#     elif polarity > 0.2:
#         return SENTIMENT_HINTS["optimism"]
#     else:
#         return SENTIMENT_HINTS["tension"]

# def merge_keyword_config(keywords):
#     """
#     Combines all the intelligence: direct matches, aliases, semantic hints, and fallback polarity.
#     Produces a full stressor configuration based on input keywords.
#     """
#     config = BASE_CONFIG.copy()

#     for word in keywords:
#         w = word.lower().strip()

#         # 1. Direct match
#         mods = KEYWORD_MAP.get(w)

#         # 2. Alias match → keyword map
#         if not mods and w in ALIASES:
#             mapped = ALIASES[w]
#             mods = KEYWORD_MAP.get(mapped) or SENTIMENT_HINTS.get(mapped)

#         # 3. Semantic or polarity fallback
#         if not mods:
#             mods = get_sentiment_boost(w)

#         # ⬆️ Apply modifications
#         for k, v in mods.items():
#             if isinstance(v, bool):
#                 config[k] = config[k] or v
#             else:
#                 config[k] += v

#     return config