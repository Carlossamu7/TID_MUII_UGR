#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Carlos Santiago Sánchez Muñoz
@date: 22 de noviembre de 2020
"""

"""
P2- AGRUPAMIENTO
"""

#############################
#####     LIBRERIAS     #####
#############################

import pandas as pd
import scipy.cluster.hierarchy as sch
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Preprocesado
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

IMPRIME_INFO = True     # Indica si imprimir información

#############################
#####     FUNCIONES     #####
#############################

""" Lectura de datos. Devuelve el dataframe.
"""
def read_data():
    names = ['Type','Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids',
             'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue','OD280/OD315 of diluted wines', 'Proline']
    df = pd.read_csv('wine.data', names=names)

    if(IMPRIME_INFO):
        print("\nInformación del dataframe:")
        print(df.info())
        print("\nDataframe:")
        print(df)
    return df

""" Divide los datos quitando la etiqueta a predecir. Devuelve X e y.
- df: dataframe.
"""
def split_data(df):
    return df.drop(columns="Type", axis=1), df["Type"]

""" Pinta el dendograma de los datos de X.
- X: datos.
- title: título. Por defecto "Dendograma".
"""
def dendograma(X, title="Dendograma"):
    print("Calculando el dendograma")
    dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
    plt.xlabel('Vinos')
    plt.ylabel('Distancias Euclídeas')
    plt.title(title)
    plt.gcf().canvas.set_window_title("Práctica 2 - Agrupamiento")
    plt.show()

""" Borra las instancias en donde alguna de las columnas sea un outlier.
- df: dataframe.
"""
def delete_outliers(df):
    df_outliers = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
    df_outliers=df_outliers.reset_index()
    del df_outliers['index']
    print("Número de outliers borrados: {}".format(len(df)-len(df_outliers)))
    return df_outliers

""" Aplica el clustering KMeans. Devuelve el clasificador y las etiquetas predichas.
- X: datos.
"""
def get_k_medias(X):
    kmeans = KMeans(n_clusters=3).fit(X)
    labels = kmeans.predict(X)
    print()
    print(kmeans)
    print("\nCentroides:")
    print(kmeans.cluster_centers_)
    print("\nEtiquetas después de kmeans:")
    print(labels)
    return kmeans, labels

""" Aplica el clustering KMeans. Devuelve el clasificador y las etiquetas predichas.
- X: datos.
- kmeans: clasificador kmedias ya entrenado.
- labels: etiquetas predichas.
"""
def plot_cluster_primera_variable(X, kmeans, labels):
    kmeans = KMeans(n_clusters=3)
    y_kmeans =kmeans.fit_predict(X.iloc[:,0:2])
    X = X.iloc[:,0:2].values

    plt.scatter(X[y_kmeans==0, 0], X[y_kmeans==0, 1], s=10, c='red', label='Clase 1')
    plt.scatter(X[y_kmeans==1, 0], X[y_kmeans==1, 1], s=10, c='blue', label='Clase 2')
    plt.scatter(X[y_kmeans==2, 0], X[y_kmeans==2, 1], s=10, c='green', label='Clase 3')
    plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=100, marker="*", c='black', label='Centroides')
    plt.legend()
    plt.title("Clusters de clases")
    plt.gcf().canvas.set_window_title("Práctica 2 - Agrupamiento")
    plt.show()


########################
#####     MAIN     #####
########################

""" Programa principal. """
def main():
    print("Leyendo el conjunto de datos")
    df = read_data()

    print("\nVariable a predecir ('Type'):")
    print(df.groupby(['Type']).size())

    print("\n----- Tratando los datos normales -----")
    X, y =split_data(df)
    dendograma(X)

    print("\n----- Tratando los datos que no tienen outliers -----")
    df_o = delete_outliers(df)
    X_o, y_o =split_data(df_o)
    dendograma(X_o, "Dendograma sin outliers")

    kmeans, labels = get_k_medias(X_o)
    # Comparamos la distribución de las clases respecto a la primera columna.
    plot_cluster_primera_variable(X_o, kmeans, labels)
    silhouette_avg3 = silhouette_score(X_o, (kmeans.labels_), metric='euclidean')
    print("\nLa puntuación de silhouette es: {}".format(silhouette_avg3))

    X_std = StandardScaler().fit_transform(X_o)
    print(X_std)

    X_pca = PCA(n_components=8).fit_transform(X_std)
    X_pca = pd.DataFrame(X_pca)
    if(IMPRIME_INFO):
        print(X_pca)
    kmeans_pca, labels_pca = get_k_medias(X_pca)
    # Comparamos la distribución de las clases respecto a la primera columna.
    #plot_cluster_primera_variable(X_pca, kmeans_pca, labels_pca)
    #silhouette_pca = silhouette_score(X_pca, (kmeans.labels_), metric='euclidean')
    #print("\nLa puntuación de silhouette es: {}".format(silhouette_pca))


if __name__ == "__main__":
	main()
