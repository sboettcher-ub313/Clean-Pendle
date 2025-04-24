# Oracle insight interpreter (rebuild marker)

import numpy as np

def interpret_forecast(pca_projection):
    first_pc = pca_projection[:, 0]
    trend = np.polyfit(range(len(first_pc)), first_pc, 1)[0]
    if trend > 0.05:
        return "🔺 Owl Pendle sees rising latent market stress.", trend
    elif trend < -0.05:
        return "🟢 Owl Pendle senses calming macro conditions.", trend
    else:
        return "⚖️ Owl Pendle observes neutral market flow.", trend