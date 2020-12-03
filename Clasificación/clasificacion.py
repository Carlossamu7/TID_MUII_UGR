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
from sklearn import svm
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_score
import statistics as stats

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
    print("Entrenando KNN")
    knn.fit(X_train, y_train)
    print('Accuracy de K-NN en el conjunto de entrenamiento: {:.2f}'.format(knn.score(X_train, y_train)))
    print('Accuracy de K-NN en el conjunto de test: {:.2f}'.format(knn.score(X_test, y_test)))
    print("Accuracys de la cross-validation (5 particiones)")
    score = cross_val_score(knn, X_train.append(X_test), y_train.append(y_test), cv=5)
    print(score)
    print("Accuracy medio de la cross-validation: {:.4f}".format(stats.mean(score)))
    print("Prediciendo etiquetas")
    return knn.predict(X_test), knn.predict_proba(X_test)

""" Entrenando con árbol de decisión.
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
    print("Accuracys de la cross-validation (5 particiones)")
    score = cross_val_score(tree, X_train.append(X_test), y_train.append(y_test), cv=5)
    print(score)
    print("Accuracy medio de la cross-validation: {:.4f}".format(stats.mean(score)))
    print("Prediciendo etiquetas")
    return tree.predict(X_test), tree.predict_proba(X_test)

""" Entrenando con SVC.
- X_train: datos de entrenamiento.
- X_test: datos de test.
- y_train: etiquetas del entrenamiento.
- y_test: etiquetas de test.
"""
def SVC(X_train, X_test, y_train, y_test):
    print("\n------ SVC ------")
    svc = svm.SVC(kernel='linear', probability=True, max_iter=500000)
    print("Entrenando SVC")
    svc.fit(X_train, y_train)
    print('Accuracy de SVM en el conjunto de entrenamiento: {:.2f}'.format(svc.score(X_train, y_train)))
    print('Accuracy de SVM en el conjunto de test: {:.2f}'.format(svc.score(X_test, y_test)))
    print("Accuracys de la cross-validation (5 particiones)")
    score = cross_val_score(svc, X_train.append(X_test), y_train.append(y_test), cv=5)
    print(score)
    print("Accuracy medio de la cross-validation: {:.4f}".format(stats.mean(score)))
    print("Prediciendo etiquetas")
    return svc.predict(X_test), svc.predict_proba(X_test)

""" Entrenando con LogisticRegression.
- X_train: datos de entrenamiento.
- X_test: datos de test.
- y_train: etiquetas del entrenamiento.
- y_test: etiquetas de test.
"""
def LR(X_train, X_test, y_train, y_test):
    print("\n------ LogisticRegression ------")
    lr = LogisticRegression()
    print("Entrenando regresión logística")
    lr.fit(X_train, y_train)
    print('Accuracy de LogisticRegression en el conjunto de entrenamiento: {:.2f}'.format(lr.score(X_train, y_train)))
    print('Accuracy de LogisticRegression en el conjunto de test: {:.2f}'.format(lr.score(X_test, y_test)))
    print("Accuracys de la cross-validation (5 particiones)")
    score = cross_val_score(lr, X_train.append(X_test), y_train.append(y_test), cv=5)
    print(score)
    print("Accuracy medio de la cross-validation: {:.4f}".format(stats.mean(score)))
    print("Prediciendo etiquetas")
    return lr.predict(X_test), lr.predict_proba(X_test)

""" Muestra matriz de confusión y un reportaje de clasificación.
- y_test: etiquetas reales.
- y_pred: etiquetas predichas.
"""
def print_plot_sol(y_test, y_pred):
    print("Matriz de confusión:")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

def plot_roc_curve(y_test, probs_knn, probs_tree, probs_svc, probs_lr):
    # keep probabilities for the positive outcome only
    probs_knn = probs_knn[:, 1]
    probs_tree = probs_tree[:, 1]
    probs_svc = probs_svc[:, 1]
    probs_lr = probs_lr[:, 1]

    # Calculamos scores
    auc_knn = roc_auc_score(y_test, probs_knn)
    auc_tree = roc_auc_score(y_test, probs_tree)
    auc_svc = roc_auc_score(y_test, probs_svc)
    auc_lr = roc_auc_score(y_test, probs_lr)

    # Los imprimimos scores
    print("KNN: ROC AUC={:.3f}".format(auc_knn))
    print("Tree: ROC AUC={:.3f}".format(auc_tree))
    print("SVC: ROC AUC={:.3f}".format(auc_svc))
    print("Logistic: ROC AUC={:.3f}".format(auc_lr))

    # Calculamos las curvas ROC
    fpr_knn, tpr_knn, _ = roc_curve(y_test, probs_knn)
    fpr_tree, tpr_tree, _ = roc_curve(y_test, probs_tree)
    fpr_svc, tpr_svc, _ = roc_curve(y_test, probs_svc)
    fpr_lr, tpr_lr, _ = roc_curve(y_test, probs_lr)

    # Dibujamos las curvas
    plt.plot(fpr_knn, tpr_knn, linestyle='--', label='KNN')
    plt.plot(fpr_tree, tpr_tree, linestyle=':', label='Tree')
    plt.plot(fpr_svc, tpr_svc, linestyle='-.', label='SVC')
    plt.plot(fpr_lr, tpr_lr, marker='.', label='Logistic')
    # Etiqueta los ejes
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.title("Curva ROC")
    plt.gcf().canvas.set_window_title("Práctica 3 - clasificación")
    plt.show()

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

    y_pred_knn, probs_knn = knn(X_train, X_test, y_train, y_test)
    print_plot_sol(y_test, y_pred_knn)
    y_pred_tree, probs_tree = decision_tree(X_train, X_test, y_train, y_test)
    print_plot_sol(y_test, y_pred_tree)
    y_pred_svc, probs_svc = SVC(X_train, X_test, y_train, y_test)
    print_plot_sol(y_test, y_pred_svc)
    y_pred_lr, probs_lr = LR(X_train, X_test, y_train, y_test)
    print_plot_sol(y_test, y_pred_lr)

    # Imprimimos las curvas ROC
    plot_roc_curve(y_test, probs_knn, probs_tree, probs_svc, probs_lr)

if __name__ == "__main__":
	main()
