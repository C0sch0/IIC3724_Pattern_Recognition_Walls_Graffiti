import pca
import feature_transformation
import feature_selection
import feature_extraction
import cv2
import sys
import numpy as np
from os import listdir
import fnmatch, os

import pandas as pd
import warnings
import tqdm
from PIL import Image
from seaborn import heatmap
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from itertools import combinations
from pybalu.feature_extraction import lbp_features, hog_features, gabor_features, haralick_features

# **Reconocimiento de Patrones en Imágenes**
# Se cuenta con una base de datos de 10.000 patches a color de 64x64 pixeles, correspondientes a porciones de paredes
# que han sido y que no han sido rayadas, distribuidas 50-50%, es decir 5.000 patches pertencientes a la
# clase 1 (rayas), y 5.000 patches pertenecientes a la clase 0 (no-rayas).
# Cada uno de estos patches cubre aproximadamente una superficie de 30cm x 30cm de la pared.
#
# Se debe disenar un clasificador que funcione con un maximo de 50 caraceristicas, para esto se deben sacar al menos 20
# caracteristica y a partir de tecnicas se seleccion o transformacion de caracteristicas se le debe proporcionar
# al clasificador un maximo de 50 caracteristicas. El clasificador a emplear es un KNN de tres vecinos.

INPUT_PATH_01 = os.path.join("Training_0")
INPUT_PATH_02 = os.path.join("Training_1")
INPUT_PATH_03 = os.path.join("Testing_0")
INPUT_PATH_04 = os.path.join("./Testing_1")


def test():
    X_Train_clean_walls = [picture for picture in listdir(INPUT_PATH_01) if picture.endswith('png')]
    x_Train_intervened_walls = [picture for picture in listdir(INPUT_PATH_02) if picture.endswith('png')]
    Y_Testing_clean_walls = [picture for picture in listdir(INPUT_PATH_03) if picture.endswith('png')]
    y_Testing_intervened_walls = [picture for picture in listdir(INPUT_PATH_04) if picture.endswith('png')]

    print("Clean Train: {0}, Intervened Train:{1},"
          " Clean Test:{2}, Intervened Test:{3}".format(len(X_Train_clean_walls),
                                                        len(x_Train_intervened_walls),
                                                        len(Y_Testing_clean_walls),
                                                        len(y_Testing_intervened_walls)))

    X0_train = extract_features('Training_0', 'png')
    X1_train = extract_features('Training_1', 'png')
    X0_test = extract_features('Testing_0', 'png')
    X1_test = extract_features('Testing_1', 'png')


    print('Training Subset:')
    X_train  = np.concatenate((X0_train, X1_train),axis=0)
    d0_train = np.zeros([X0_train.shape[0], 1],dtype=int)
    d1_train = np.ones([X1_train.shape[0], 1],dtype=int)
    d_train  = np.concatenate((d0_train, d1_train),axis=0)
    print('Original extracted features: '+str(X_train.shape[1])+ '('+str(X_train.shape[0])+' samples)')


    # Eliminamos caracteristicas altamente correlacionadas o constantes para set de Training
    sclean = clean(X_train,show=True)
    X_train_clean = X_train[:,sclean]
    print('cleaned features: '+str(X_train_clean.shape[1])+ '('+str(X_train_clean.shape[0])+' samples)')


    # Normalizamos las columnas de datos
    X_train_norm = X_train_clean * (1 / X_train_clean.std(0)) + (- X_train_clean.mean(0) / X_train_clean.std(0))

    print('normalized features: '+str(X_train_norm.shape[1])+ '('+str(X_train_norm.shape[0])+' samples)')


    # Training: Feature selection
    ssfs = Sequential_Feature_Selector(X_train_norm, d_train, n_features=80)
    X_train_sfs = X_train_norm[:,ssfs]
    print('selected features: '+str(X_train_sfs.shape[1])+ '('+str(X_train_sfs.shape[0])+' samples)')


    # Testing dataset
    print('Testing Subset:')
    X_test  = np.concatenate((X0_test,X1_test),axis=0)
    d0_test = np.zeros([X0_test.shape[0],1],dtype=int)
    d1_test = np.ones([X1_test.shape[0],1],dtype=int)
    d_test  = np.concatenate((d0_test,d1_test),axis=0)

    # Testing: Cleaning
    X_test_clean = X_test[:,sclean]

    # Testing: Normalization
    X_test_norm = X_test_clean * (1 / X_train_clean.std(0)) + (- X_train_clean.mean(0) / X_train_clean.std(0))

    # Testing: Feature selection
    X_test_sfs = X_test_norm[:,ssfs]
    print('clean+norm+sfs features: '+str(X_test_sfs.shape[1])+ '('+str(X_test_sfs.shape[0])+' samples)')


    # Classification on Testing dataset
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train_sfs, d_train)

    acc = accuracy_score(d_test,knn.predict(X_test_sfs))
    print('Confusion Matrix:')
    print(confusion_matrix(d_test,knn.predict(X_test_sfs)) )
    print('Accuracy = '+str(acc))
    print()


    plt.figure(figsize=(7,5))
    heatmap(confusion_matrix(d_test,knn.predict(X_test_sfs)) , annot=True, fmt="d", cmap="YlGnBu")
    plt.xlim(0,confusion_matrix(d_test,knn.predict(X_test_sfs)).shape[0])
    plt.ylim(confusion_matrix(d_test,knn.predict(X_test_sfs)).shape[0],0)
    plt.title('Confusion Matrix Testing',fontsize=14)
    plt.show()
