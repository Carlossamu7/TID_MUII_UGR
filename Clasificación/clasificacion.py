#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Carlos Santiago Sánchez Muñoz
@date: 27 de noviembre de 2020
"""

"""
P3- CLASIFICACION
"""

#############################
#####     LIBRERIAS     #####
#############################

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

IMPRIME_INFO = True     # Indica si imprimir información

#############################
#####     FUNCIONES     #####
#############################

""" Lectura de datos. Devuelve el dataframe.
"""
def read_data():
    df = pd.read_excel('eBayAuctions.xls')

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
    return df.drop(columns="Competitive?", axis=1), df["Competitive?"]

""" Entrenando con k vecinos más cercanos
- X_train: datos de entrenamiento.
- X_test: datos de test.
- y_train: etiquetas del entrenamiento.
- y_test: etiquetas de test.
"""
def knn(X_train, X_test, y_train, y_test):
    knn = KNeighborsClassifier(5)
    knn.fit(X_train, y_train)
    print('Accuracy de K-NN en el conjunto de entrenamiento: {:.2f}'.format(knn.score(X_train, y_train)))
    print('Accuracy de K-NN en el conjunto de test: {:.2f}'.format(knn.score(X_test, y_test)))
    return knn.predict(X_test)

""" Muestra matriz de confusión.
- y_real: etiquetas reales.
- y_pred: etiquetas predichas.
- message: mensaje que complementa la matriz de confusión.
- norm (op): indica si normalizar (dar en %) la matriz de confusión. Por defecto 'True'.
"""
def show_confussion_matrix(y_real, y_pred, message="", norm=True):
	mat = confusion_matrix(y_real, y_pred)
	if(norm):
		mat = 100*mat.astype("float64")/mat.sum(axis=1)[:, np.newaxis]
	fig, ax = plt.subplots()
	ax.matshow(mat, cmap="GnBu")
	ax.set(title="Matriz de confusión {}".format(message),
		   xticks=np.arange(3), yticks=np.arange(3),
		   xlabel="Etiqueta", ylabel="Predicción")

	for i in range(3):
		for j in range(3):
			if(norm):
				ax.text(j, i, "{:.0f}%".format(mat[i, j]), ha="center", va="center",
					color="black" if mat[i, j] < 50 else "white")
			else:
				ax.text(j, i, "{:.0f}".format(mat[i, j]), ha="center", va="center",
					color="black" if mat[i, j] < 50 else "white")
	plt.gcf().canvas.set_window_title("Práctica 1 - Preprocesamiento")
	plt.show()

""" Muestra matriz de confusión y un reportaje de clasificación.
- y_test: etiquetas reales.
- y_pred: etiquetas predichas.
"""
def print_plot_sol(y_test, y_pred):
    print(confusion_matrix(y_test, y_pred))
    show_confussion_matrix(y_test, y_pred, norm=False)
    show_confussion_matrix(y_test, y_pred)
    print(classification_report(y_test, y_pred))

########################
#####     MAIN     #####
########################

""" Programa principal. """
def main():
    print("Leyendo el conjunto de datos")
    df = read_data()

    X, y = split_data(df)

    df.hist()
    plt.title("Gráficas")
    plt.gcf().canvas.set_window_title("Práctica 3 - clasificación")
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    y_pred_knn = knn(X_train, X_test, y_train, y_test)
    print_plot_sol(y_test, y_pred_knn)


if __name__ == "__main__":
	main()
