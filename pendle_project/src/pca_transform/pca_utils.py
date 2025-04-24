# PCA utilities (rebuild marker)

from sklearn.decomposition import PCA

def fit_pca(df, n_components=3):
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(df)
    return pca, transformed