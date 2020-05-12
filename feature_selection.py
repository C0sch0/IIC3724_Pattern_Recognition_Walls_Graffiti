from imageio import imread as _imread
import cv2
import sys
import numpy as np
from os import listdir
import fnmatch, os

def imread(filename, *, normalize=False, flatten=False):
    img = _imread(filename)
    if flatten:
        img = img @ [0.299, 0.587, 0.114]
    if normalize:
        return (img / 255)
    return img.astype(_np.uint8)

# Para aprender como usar SFS, vimos el codigo base de pybalu, pero reemplazamos todas las dependencias
# dentro de la libreria por metodos propios.

def Sequential_Feature_Selector(features, classes, n_features):
    remaining_feats = set(np.arange(features.shape[1]))
    selected = list()
    curr_feats = np.zeros((features.shape[0], 0))
    options = dict()

    def fisher_score(i):
        feats = np.hstack([curr_feats, features[:, i].reshape(-1, 1)])
        m = features.shape[1]
        norm = classes.ravel() - classes.min()
        max_class = norm.max() + 1
        p = np.ones(shape=(max_class, 1)) / max_class

        features_mean = feats.mean(0)
        cov_w = np.zeros(shape=(m, m))
        cov_b = np.zeros(shape=(m, m))

        for k in range(max_class):
            ii = (norm == k)
            class_features = features[ii, :]
            class_mean = class_features.mean(0)
            class_cov = np.cov(class_features, rowvar=False)

            cov_w += p[k] * class_cov

            dif = (class_mean - features_mean).reshape((m, 1))
            cov_b += p[k] * dif @ dif.T
        try:
            return np.trace(np.linalg.inv(cov_w) @ cov_b)
        except np.linalg.LinAlgError:
            return - np.inf

    _range = tqdm.trange(n_features, desc='', unit_scale=True, unit=' features')

    for _ in _range:
        new_selected = max(remaining_feats, key=fisher_score)
        selected.append(new_selected)
        remaining_feats.remove(new_selected)
        curr_feats = np.hstack([curr_feats, features[:, new_selected].reshape(-1, 1)])

    return np.array(selected)