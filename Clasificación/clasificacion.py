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
from sklearn.tree import DecisionTreeClassifier

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

""" Convierte a int la variable 'endDay'.
- df: dataframe.
"""
def int_endDay(df):
    list = []
    for i in range(len(df)):
        if(df["endDay"][i]=="Sun"):
            list.append(0)
        elif(df["endDay"][i]=="Mon"):
            list.append(1)
        elif(df["endDay"][i]=="Tue"):
            list.append(2)
        elif(df["endDay"][i]=="Wed"):
            list.append(3)
        elif(df["endDay"][i]=="Thu"):
            list.append(4)
        elif(df["endDay"][i]=="Fri"):
            list.append(5)
        elif(df["endDay"][i]=="Sat"):
            list.append(6)
    df = df.drop("endDay", axis=1)
    df["endDay"] = list
    return df

""" Convierte a int la variable 'currency'.
- df: dataframe.
"""
def int_currency(df):
    list = []
    for i in range(len(df)):
        if(df["currency"][i]=="US"):
            list.append(0)
        elif(df["currency"][i]=="EUR"):
            list.append(1)
        elif(df["currency"][i]=="GBP"):
            list.append(2)
    df = df.drop("currency", axis=1)
    df["currency"] = list
    return df

""" Convierte a int la variable 'Category'.
- df: dataframe.
"""
def int_Category(df):
    list = []
    for i in range(len(df)):
        if(df["Category"][i]=="Antique/Art/Craft"):
            list.append(0)
        elif(df["Category"][i]=="Automotive"):
            list.append(1)
        elif(df["Category"][i]=="Books"):
            list.append(2)
        elif(df["Category"][i]=="Business/Industrial"):
            list.append(3)
        elif(df["Category"][i]=="Clothing/Accessories"):
            list.append(4)
        elif(df["Category"][i]=="Coins/Stamps"):
            list.append(5)
        elif(df["Category"][i]=="Collectibles"):
            list.append(6)
        elif(df["Category"][i]=="Computer"):
            list.append(7)
        elif(df["Category"][i]=="Electronics"):
            list.append(8)
        elif(df["Category"][i]=="EverythingElse"):
            list.append(9)
        elif(df["Category"][i]=="Health/Beauty"):
            list.append(10)
        elif(df["Category"][i]=="Home/Garden"):
            list.append(11)
        elif(df["Category"][i]=="Jewelry"):
            list.append(12)
        elif(df["Category"][i]=="Music/Movie/Game"):
            list.append(13)
        elif(df["Category"][i]=="Photography"):
            list.append(14)
        elif(df["Category"][i]=="Pottery/Glass"):
            list.append(15)
        elif(df["Category"][i]=="SportingGoods"):
            list.append(16)
        elif(df["Category"][i]=="Toys/Hobbies"):
            list.append(17)
    df = df.drop("Category", axis=1)
    df["Category"] = list
    return df

""" Convierte a int las variables categóricas.
- df: dataframe.
"""
def to_int_categorical(df):
    df = int_endDay(df)
    df = int_currency(df)
    df = int_Category(df)
    return df

""" Entrenando con k vecinos más cercanos.
- X_train: datos de entrenamiento.
- X_test: datos de test.
- y_train: etiquetas del entrenamiento.
- y_test: etiquetas de test.
"""
def knn(X_train, X_test, y_train, y_test):
    print("\n------ KNN ------")
    knn = KNeighborsClassifier(5)
    print("Entrenando knn")
    knn.fit(X_train, y_train)
    print('Accuracy de K-NN en el conjunto de entrenamiento: {:.2f}'.format(knn.score(X_train, y_train)))
    print('Accuracy de K-NN en el conjunto de test: {:.2f}'.format(knn.score(X_test, y_test)))
    print("Prediciendo etiquetas")
    return knn.predict(X_test)

""" Entrenando con k vecinos más cercanos
- X_train: datos de entrenamiento.
- X_test: datos de test.
- y_train: etiquetas del entrenamiento.
- y_test: etiquetas de test.
"""
def decision_tree(X_train, X_test, y_train, y_test):
    print("\n------ DecisionTreeClassifier ------")
    tree = DecisionTreeClassifier(max_depth=10)
    print("Entrenando el árbol de decisión")
    tree = tree.fit(X_train, y_train)
    print('Accuracy de DecisionTreeClassifier en el conjunto de entrenamiento: {:.2f}'.format(tree.score(X_train, y_train)))
    print('Accuracy de DecisionTreeClassifier en el conjunto de test: {:.2f}'.format(tree.score(X_test, y_test)))
    print("Prediciendo etiquetas")
    return tree.predict(X_test)

""" Muestra matriz de confusión y un reportaje de clasificación.
- y_test: etiquetas reales.
- y_pred: etiquetas predichas.
"""
def print_plot_sol(y_test, y_pred):
    print("Matriz de confusión:")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

########################
#####     MAIN     #####
########################

""" Programa principal. """
def main():
    print("Leyendo el conjunto de datos")
    df = read_data()

    # Preprocesando las variables categóricas
    df = to_int_categorical(df)

    if(IMPRIME_INFO):
        for atr in df:
            print()
            print(df.groupby([atr]).size())
    df.hist()
    plt.gcf().canvas.set_window_title("Práctica 3 - clasificación")
    plt.show()

    # Dividiendo datos en input/output
    X, y = split_data(df)
    # Dividiendo datos en train y test
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    y_pred_knn = knn(X_train, X_test, y_train, y_test)
    print_plot_sol(y_test, y_pred_knn)
    y_pred_tree = decision_tree(X_train, X_test, y_train, y_test)
    print_plot_sol(y_test, y_pred_tree)


if __name__ == "__main__":
	main()
