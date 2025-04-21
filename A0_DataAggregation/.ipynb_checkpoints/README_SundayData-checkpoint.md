
# 🧠 SundayData: Mapping Economic Indicators & Market Signals

SundayData is a visual data narrative project that explores how macroeconomic indicators, market sentiment, and social search behavior correlate with—and sometimes precede—market volatility. Using Yahoo Finance, FRED, and Google Trends data from 2000 to present, we explore the emotional undercurrent beneath financial systems.

> “Even after checking the rearview and side mirrors, blind spots remain.”  
> This project exists to map those blind spots using data, visuals, and intuition.

---

## 📦 Project Structure

```
SundayData/
│
├── data/
│   ├── raw/           # Original datasets (e.g., NAVAll.txt, missing summaries)
│   ├── processed/     # Cleaned CSVs, rolling indicators, lagged metrics
│   └── external/      # India mutual funds, WSB stocks, startup data
│
├── notebooks/
│   ├── SundayData.ipynb
│   ├── EDA_2.ipynb
│   ├── EDA.ipynb
│   ├── Overview.ipynb
│   ├── Notes.ipynb
│   ├── About.ipynb
│   ├── Notebook_0_Data_Collection.ipynb
│   └── Notebook_0_Data_Collection copy.ipynb
│
├── images/
│   ├── outputs/       # Final plots from notebooks
│   └── references/    # Annotated screenshots, memes, source overlays
│
├── scripts/
│   └── (Optional: loaders, preprocessors, visual generators)
│
└── README.md
```

---

## 📊 Key Questions Explored

- Is volatility structurally higher in post-COVID markets?
- Does VIX still lead Nasdaq volatility—or have their roles diverged?
- How do investor sentiments (Google Trends) map onto crash events?
- Can lagged GDP, sentiment, or inflation data serve as early warnings?

---

## 📈 Sample Visuals (from `images/outputs/`)

- VolatilityChart.png
- SP500PriceTrend.png
- interest rate inflation.png
- macroindicators_timeseries.png
- lagging_gdp_market_trends.png

---

## 🧠 Methods & Inspiration

- Finance: Lagged indicators, volatility clustering
- ML: Anomaly detection and signal compounding
- Behavioral Econ: Interpreting search interest as emotional market states
- Diagnostic AI: Synthesizing weak signals like in radiology or ICU alerts

---

## ⚠️ Notes

- Some earlier files may have expired — reupload for scripting support.
- Want an auto-gallery, loader script, or publishing-ready report? Just ask.

---
