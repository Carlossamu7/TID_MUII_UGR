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


if __name__ == "__main__":
	main()
