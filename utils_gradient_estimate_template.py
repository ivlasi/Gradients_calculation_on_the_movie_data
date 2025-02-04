from brainspace.gradient import GradientMaps
import numpy as np
import matplotlib.pyplot as plt

def btftd_util_gradients_estimate_template(cfg, S):
    """
    Calculate a Gradient Embedding Template for a group/set of subjects.

    Parameters:
    cfg : dict
        Configuration with settings such as typeAverage, doPlots, etc.
    S : dict
        Subject-specific information like parcellation mappings.

    Returns:
    GradMap : dict
        Contains calculated gradients for different kernels.
    """
    typeAverage = cfg.get("typeAverage", "pca")  # mean | pca | median
    doPlots = cfg.get("doPlots", 0)
    embedding = cfg.get("embeddings", "diffusionEmbedding")  # Case-insensitive options
    doFisherZ = cfg.get("doFisherZ", 0)
    parcellation = cfg.get("parcellation", None)  # Parcelation mapping required for visualization
    ncomp = cfg.get("ncomp", 10)  # Number of gradients to estimate
    doZscoreGrad = cfg.get("doZscore", 0)  # Whether to zscore each gradient
    
    GradMap = {
        "CS": GradientMaps(kernel="cosine", approach=embedding, n_components=ncomp),
        "NA": GradientMaps(kernel="normalized_angle", approach=embedding, n_components=ncomp),
        "G": GradientMaps(kernel="gaussian", approach=embedding, n_components=ncomp),
        "P": GradientMaps(kernel="pearson", approach=embedding, n_components=ncomp),
        "SM": GradientMaps(kernel="spearman", approach=embedding, n_components=ncomp),
    }
    Rtemp = cfg["conmat"]

    if doFisherZ:
        Rtemp = np.arctanh(Rtemp)  # Apply Fisher Z-transform

    if typeAverage == "mean":
        Rtemp = np.nanmean(Rtemp, axis=2)
    elif typeAverage == "median":
        Rtemp = np.nanmedian(Rtemp, axis=2)
    elif typeAverage == "pca":
        from sklearn.decomposition import PCA

        Rtemp = np.transpose(Rtemp, (2, 0, 1))  # Permute dimensions
        idxcm = np.triu_indices(Rtemp.shape[1], k=1)  # Upper triangular indices
        Rtemp = Rtemp[:, idxcm[0], idxcm[1]].reshape(-1, len(idxcm[0]))
        Rtemp[np.isnan(Rtemp)] = np.finfo(float).eps  # Replace NaNs with small values

        pca = PCA(n_components=1)
        scores = pca.fit_transform(Rtemp)
        Rt = np.zeros((nparcel, nparcel))
        Rt[idxcm] = scores.ravel()
        Rt += Rt.T
        Rtemp = Rt
    tmp = Rtemp
    np.fill_diagonal(tmp, 1)  # Ensure diagonal is 1
    for key in GradMap:
        GradMap[key].fit(tmp)
    if doPlots:
        # Example: Creating heatmaps for gradients
        for key, gmap in GradMap.items():
            gradients = gmap.gradients_[0]  # Assuming `gradients_` stores the gradients
            plt.imshow(gradients, cmap="RdBu_r", interpolation="nearest")
            plt.title(f"Gradient: {key}")
            plt.colorbar()
            plt.show()
    return GradMap
