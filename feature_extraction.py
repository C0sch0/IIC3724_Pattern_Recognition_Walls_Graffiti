import pandas as pd
import warnings
import tqdm
import cv2
import sys
import numpy as np
from os import listdir
import fnmatch, os
from PIL import Image
from seaborn import heatmap
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from itertools import combinations
from pybalu.feature_extraction import lbp_features, hog_features, gabor_features, haralick_features

# Extraccion de caracteristicas
# Probamos con Gabor, Haralich, HOG, LBP, LBP por gama de colores y escala de grises

def imshow(image):
    pil_image = Image.fromarray(image)
    pil_image.show()


def get_image(path, show=False):
    img = cv2.imread(path)
    if show:
        imshow(img)
    return img


def dirfiles(img_path, img_ext):
    img_names = fnmatch.filter(sorted(os.listdir(img_path)), img_ext)
    return img_names


def num2fixstr(x, d):
    st = '%0*d' % (d, x)
    return st


def extract_features(dirpath, fmt):
    st = '*.' + fmt
    img_names = dirfiles(dirpath + '/', st)
    n = len(img_names)
    print(n)
    for i in range(n):
        img_path = img_names[i]
        img = get_image(dirpath + '/' + img_path)
        escala_grises = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        X_0 = lbp_features(escala_grises, hdiv=1, vdiv=1, mapping='nri_uniform')
        rojo = lbp_features(img[:, :, 0], hdiv=1, vdiv=1, mapping='nri_uniform')
        verde = lbp_features(img[:, :, 1], hdiv=1, vdiv=1, mapping='nri_uniform')
        azul = lbp_features(img[:, :, 2], hdiv=1, vdiv=1, mapping='nri_uniform')
        # Haralick = haralick_features(img.astype(int))
        # hog = hog_features(i, v_windows=3, h_windows=3, n_bins=8)
        features = np.asarray(np.concatenate((X_0, rojo, verde, azul)))

        if i == 0:
            m = features.shape[0]
            data = np.zeros((n, m))
            print('size of extracted features:')
            print(features.shape)
        data[i] = features
    return data
