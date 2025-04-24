import os
import streamlit as st

# Local import
from components.forecast_demo import run_forecast_logic

# 🔧 UI Setup
st.set_page_config(page_title="Pendle the Pixel Oracle", page_icon="🧠")
st.title("🦉 Pendle the Owl Oracle")
st.caption("Forecasting macro mood through latent winds...")

# 🧠 User Keyword Input
keywords_input = st.text_input("What themes are on your mind? (comma separated)", "war, layoffs, AI hype")
keywords = [w.strip() for w in keywords_input.split(",") if w.strip()]

# 🔮 Main interaction
if st.button("🔮 Ask Pendle"):    
    insight, trend, rare_rate, pr_auc, config, fig = run_forecast_logic(keywords)

    # 🖼️ Pick Owl Sprite based on mood
    current_dir = os.path.dirname(__file__)
    asset_path = os.path.join(current_dir, "assets")

    if trend > 0.05:
        sprite_file = os.path.join(asset_path, "owl_alert.png")
    elif trend < -0.05:
        sprite_file = os.path.join(asset_path, "owl_calm.png")
    else:
        sprite_file = os.path.join(asset_path, "owl_idle.png")

    try:
        st.image(sprite_file, width=120)
    except Exception as e:
        st.warning(f"Could not load owl sprite from: {sprite_file}")
        st.exception(e)

    # 🧾 Output
    st.markdown(f"### {insight}")
    st.markdown(f"**Rare Event Rate:** `{rare_rate:.3f}`")
    st.markdown(f"**PR AUC:** `{pr_auc:.3f}`")
    st.markdown("#### Generator Configuration Used:")
    st.json(config)

    st.pyplot(fig)


##################
### CHECKPOINT ###
##################
# import os
# import streamlit as st
# from components.forecast_demo import run_forecast_logic

# # Get correct directory path for assets
# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# ASSETS_DIR = os.path.join(CURRENT_DIR, "assets")

# st.set_page_config(page_title="Pendle the Pixel Oracle", page_icon="🧠")
# st.title("🦉 Pendle the Owl Oracle")
# st.caption("Forecasting macro mood through latent winds...")

# keywords = st.text_input("What themes are on your mind? (comma separated)", "war, layoffs, AI hype")
# keywords = [w.strip() for w in keywords.split(",")]

# if st.button("🔮 Ask Pendle"):
#     insight, trend, rare_rate, pr_auc, config, fig = run_forecast_logic(keywords)

#     if trend > 0.05:
#         sprite_path = os.path.join(ASSETS_DIR, "owl_alert.png")
#     elif trend < -0.05:
#         sprite_path = os.path.join(ASSETS_DIR, "owl_calm.png")
#     else:
#         sprite_path = os.path.join(ASSETS_DIR, "owl_idle.png")

#     st.image(sprite_path, width=120)
#     st.markdown(f"### {insight}")
#     st.markdown(f"**Rare Event Rate:** {rare_rate:.3f}")
#     st.markdown(f"**PR AUC:** {pr_auc:.3f}")
#     st.markdown("#### Generator Config:")
#     st.json(config)
#     st.pyplot(fig)