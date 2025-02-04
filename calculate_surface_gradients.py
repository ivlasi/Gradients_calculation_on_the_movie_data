import numpy as np
import os
from scipy.stats import zscore
from utils_gradient_estimate_template import btftd_util_gradients_estimate_template
from brainspace.gradient import GradientMaps

# Define a function to calculate gradients using surface-based connectivity
def btftd_01_proc_gradiens_surf(S, T, cfg):
    """
    Calculate gradients using surface-based connectivity.
    Based on MATLAB script in Bethlehem et al. 2019.
    """

    # Set path to functional surface data

    # Flags to control processing steps
    # doConnectomesCamcan = False
    # doConnectomesGenfi = False
    # doLoadConnectomesCamcan = False
    # doLoadConnectomesGenfi = False

    # Extract configuration parameters
    # parcellation = cfg['parcellation']
    # doPlots = cfg['doPlots']
    # doGroupTemplate = cfg['doGroupTemplate']
    kerneltype = cfg['kerneltype']
    doZscoreGrad = cfg['doZscoreGrad']
    doFisherZ = cfg['doFisherZ']
    # Grad = cfg['Grad']

        # Define parcellation settings
    # if parcellation == 'sjh':
    #     parc = S['parcellation']['sjh']['labels']
    #     uparcel = np.unique(parc)
    #     nparcel = len(uparcel)
    # elif parcellation == 'hcp':
    #     parc = S['parcellation']['hcp']['labels']
    #     uparcel = np.unique(parc[parc > 0])  # Non-zero parcels only
    #     nparcel = len(uparcel)
    # elif parcellation == 'bn':  # Brainnetome Atlas
    #     parc = S['parcellation']['bn']['labels']
    #     uparcel = np.unique(parc[parc > 0])  # Non-zero parcels only
    #     nparcel = len(uparcel)
    #     Placeholder for individual gradient calculations
    # GradMap = Grad['holdOut']
    # since the data is already parcelated and we dont have S or T we just give the size of the conmat
    nparcel = cfg.get("nparcel", 349)
    conmat = cfg['conmat']
    holdOut = cfg['holdOut']
    ncompAll = holdOut.shape[1]
    ncomp = cfg.get("ncomp", 3)
    embedding = cfg.get("embeddings", "diffusionEmbedding")
    
    # conmat.shape[2] is the number of subjects
    explainedCorr = np.empty((conmat.shape[2], ncomp))
    explainedSVD = np.empty((conmat.shape[2], ncomp))
    explainedXFM = np.empty(conmat.shape[2])
    alignEmbed = np.empty((nparcel, ncompAll, conmat.shape[2]))
    corrGrad = np.empty((conmat.shape[2], ncomp*2))

    if doZscoreGrad:
        holdOut = zscore(holdOut)

    # Parallel loop for each subject
    for isub in range(conmat.shape[2]):
        r = conmat[:, :, isub]
        if doFisherZ:
            r = np.arctanh(r)
        r[np.eye(nparcel, dtype=bool)] = 1  # Set diagonal to 1
        r[np.isnan(r)] = np.finfo(float).eps  # Replace NaNs with epsilon
        GradMap = GradientMaps(kernel=kerneltype, approach=embedding, n_components=ncomp)
        # Compute gradients using fit method (to be implemented)
        # print(r.shape)
        GradMap.fit(r)
        lambdas = GradMap.lambdas_
        tempEmbed = GradMap.gradients_
        indiEmbed = tempEmbed

        if doZscoreGrad:
            indiEmbed = zscore(indiEmbed)

        # calculate the correlation between the holdout and individual embeddings
        corr_matrix = np.corrcoef(indiEmbed[:, :ncomp].T, holdOut[:, :ncomp].T)
        # for icomp = 1:ncomp
        #     [~, idx] = max(abs(r(ncomp+icomp,1:ncomp)));
        #     iVar(icomp) = lambdas(idx)/sum(lambdas);
        # end
        explainedCorr[isub, :] = np.diag(corr_matrix[:ncomp, ncomp:])


        # Gradient alignment and explained variance
        U, S0, Vt = np.linalg.svd(np.dot(holdOut.T, indiEmbed), full_matrices=False)
        # print(S0.shape)
        # lambdas = np.diag(S0)
        lambdas = S0
        # print(lambdas.shape)
        explainedSVD[isub, :] = lambdas / np.sum(lambdas)
        
        # min_cols = min(U.shape[1], Vt.shape[0])
        # U = U[:, :min_cols]
        # Vt = Vt[:min_cols, :]

        xfms = np.dot(Vt.T, U.T)
        explainedXFM[isub] = np.rad2deg(np.arccos((np.trace(xfms) - 1) / 2))
        alignEmbed[:, :, isub] = np.dot(indiEmbed, xfms)

        # Calculate explained correlation for gradients
        # print(alignEmbed.shape)
        # print(alignEmbed)
        # print(holdOut.shape)
        # print(holdOut)
        corrGrad[isub] = np.diag(np.corrcoef(alignEmbed[:, :ncomp, isub], holdOut[:, :ncomp], rowvar=False))
        # explainedCorr[isub, :] = corrGrad
    return explainedCorr, explainedSVD, explainedXFM, alignEmbed, corrGrad

