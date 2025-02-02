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
import numpy as np
import scipy.cluster.hierarchy as sch
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix
from scipy import stats

# Preprocesado
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# DBSCAN
from sklearn.cluster import DBSCAN

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
    print(kmeans)
    print("\nCentroides:")
    print(kmeans.cluster_centers_)
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

    # CLUSTERING JERÁRQUICO
    print("\n----- Tratando los datos normales -----")
    X, y =split_data(df)
    dendograma(X)

    # BORRANDO OUTLIERS
    print("\n----- Tratando los datos que no tienen outliers -----")
    df_o = delete_outliers(df)
    X_o, y_o =split_data(df_o)
    dendograma(X_o, "Dendograma sin outliers")

    # KMEANS
    print("\n----- K-MEANS -----")
    kmeans, labels = get_k_medias(X_o)
    # Comparamos la distribución de las clases respecto a la primera columna.
    plot_cluster_primera_variable(X_o, kmeans, labels)
    # Cambiamos etiquetas para ajustar el cluster. 0->3
    for i in range(len(labels)):
        if labels[i]==0: labels[i]=3
        elif labels[i]==1: labels[i]=1
        elif labels[i]==2: labels[i]=2
    print("\nEtiquetas predichas por KMEANS:")
    print(labels)
    print("\nEtiquetas reales:")
    print(np.array(y_o))
    print("\nNOTA: El algoritmo KMEANS tiene un gran éxito en la clasificación")
    silhouette = silhouette_score(X_o, (kmeans.labels_), metric='euclidean')
    aciertos = 0
    for i in range(len(labels)):
        if labels[i]==y_o[i]:
            aciertos = aciertos+1
    print("\nEl accuracy es: {}".format(round(aciertos/len(labels),3)))
    print("\nLa puntuación de silhouette es: {}".format(round(silhouette,3)))

    # PREPROCESANDO (StandardScaler + PCA)
    print("\n----- PREPROCESANDO (StandardScaler + PCA) -----")
    X_std = StandardScaler().fit_transform(X_o)
    X_pca = PCA(n_components=8).fit_transform(X_std)
    X_pca = pd.DataFrame(X_pca)
    if(IMPRIME_INFO):
        print("Conjunto de datos después del preprocesamiento:")
        print(X_pca)
    kmeans_pca, labels_pca = get_k_medias(X_pca)
    labels_pca = np.array(labels_pca)
    # Comparamos la distribución de las clases respecto a la primera columna.
    plot_cluster_primera_variable(X_pca, kmeans_pca, labels_pca)
    # Cambiamos etiquetas para ajustar el cluster. 0->3
    for i in range(len(labels_pca)):
        if labels_pca[i]==0: labels_pca[i]=3
        elif labels_pca[i]==1: labels_pca[i]=1
        elif labels_pca[i]==2: labels_pca[i]=2
    print("\nEtiquetas predichas por KMEANS:")
    print(labels_pca)
    print("\nEtiquetas reales:")
    print(np.array(y_o))
    print("\nNOTA: El algoritmo KMEANS sobre los datos preprocesados tiene un gran éxito en la clasificación")
    silhouette_pca = silhouette_score(X_pca, labels_pca, metric='euclidean')
    aciertos = 0
    for i in range(len(labels_pca)):
        if labels_pca[i]==y_o[i]:
            aciertos = aciertos+1
    print("\nEl accuracy es: {}".format(round(aciertos/len(labels_pca),3)))
    print("\nLa puntuación de silhouette es: {}".format(round(silhouette_pca)))

    # DBSCAN
    print("\n----- DBSCAN -----")
    db = DBSCAN(eps=2, min_samples=3).fit(X_pca)
    print(db)
    labs = np.array(db.labels_)
    # Hacemos la asignación de labels para comparar con la primera columna:
    # 2->3 1->4 0->1 -1->2
    for i in range(len(labs)):
        if labs[i]==0: labs[i]=1
        elif labs[i]==2: labs[i]=3
        elif labs[i]==-1: labs[i]=2
        elif labs[i]==1: labs[i]=4
    print("Etiquetas predichas por DBSCAN:")
    print(labs)
    print("\nEtiquetas reales:")
    print(np.array(y_o))
    print("\nMatriz de confusión:")
    print(confusion_matrix(y_o, labs))
    print("NOTA: se observan algunos outliers (cuarta clase)")


if __name__ == "__main__":
	main()
