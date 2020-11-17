#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Carlos Santiago Sánchez Muñoz
@date: 1 de noviembre de 2020
"""

############################
###   Preprocesamiento   ###
############################

#############################
#####     LIBRERIAS     #####
#############################

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Para imprimir el árbol
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus

IMPRIME_INFO = False

def read_data(mini=True):
    if(mini):
        df = pd.read_excel(r'accidentes_mini.xls', sheet_name='datos')
    else:
        df = pd.read_excel(r'accidentes.xls', sheet_name='datos')

    df_I = df.copy()
    df = df.drop(columns = ['WKDY_I', 'HOUR_I', 'MANCOL_I', 'RELJCT_I', 'ALIGN_I', 'PROFIL_I', 'SURCON_I', 'TRFCON_I', 'SPDLIM_H', 'LGTCON_I', 'WEATHR_I', 'ALCHL_I'])
    df_I = df_I.drop(columns = ['WEEKDAY', 'HOUR', 'MAN_COL', 'REL_JCT', 'ALIGN', 'PROFILE', 'SUR_COND', 'TRAF_CON', 'SPD_LIM', 'LGHT_CON', 'WEATHER', 'ALCOHOL'])
    return df, df_I

def construct_class_variable(df):
    df['CRASH_TYPE'] = df['INJURY_CRASH'] + 2*df['FATALITIES']
    return df.drop(columns = ['PRPTYDMG_CRASH', 'INJURY_CRASH', 'FATALITIES'])

def train_test_data(df, feature_cols, label):
    X = df[feature_cols]
    y = df[label]
    return train_test_split(X, y, test_size=0.2, random_state=1) # 80% training Y 20% test

def summarize_info(X_train, X_test, y_train, y_test, title=""):
    print("\n------ Información del conjunto de datos " + title + "------")
    print("Tamaño de X_train: {}".format(X_train.shape))
    print("Tamaño de X_test: {}".format(X_test.shape))
    print("Tamaño de y_train: {}".format(y_train.shape))
    print("Tamaño de y_test: {}\n".format(y_test.shape))

def draw_png(clf, feature_cols, title="arbol.png"):
    dot_data = StringIO()
    export_graphviz(clf,
                    out_file=dot_data,
                    filled=True,
                    rounded=True,
                    special_characters=True,
                    feature_names=feature_cols,
                    class_names=['0','1','2'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png(title)
    Image(graph.create_png())


print("Leyendo el conjunto de datos")
df, df_I = read_data()

if(IMPRIME_INFO):
    print(df.info())
    print(df_I.info())
    #for x in df.columns:
    #    print(df.groupby([x]).size())
    #for x in df_I.columns:
    #    print(df_I.groupby([x]).size())

print("\n--> Variables sobre las que se va construir la etiqueta a predecir:")
print(df.groupby(['PRPTYDMG_CRASH', 'INJURY_CRASH', 'FATALITIES']).size())

print("\nTamaño antes de construir la variable predecir: {}".format(df.shape))
df = construct_class_variable(df)
df_I = construct_class_variable(df_I)
print("Tamaño después de construir la variable predecir: {}".format(df.shape))

if(IMPRIME_INFO):
    print("\n--> Columnas por el momento:")
    print(df.columns)
    print(df_I.columns)

print("\n--> Etiqueta a predecir:")
print(df.groupby(['CRASH_TYPE']).size())
#print(df_I.groupby(['CRASH_TYPE']).size())

# Conjuntos de entrenamiento y test
feature_cols = ['MONTH', 'WEEKDAY', 'HOUR', 'VEH_INVL', 'NON_INVL', 'LAND_USE', 'MAN_COL', 'INT_HWY', 'REL_JCT',
                'REL_RWY', 'TRAF_WAY', 'NUM_LAN', 'ALIGN', 'PROFILE', 'SUR_COND', 'TRAF_CON', 'SPD_LIM', 'LGHT_CON',
                'WEATHER', 'SCHL_BUS', 'PED_ACC', 'ALCOHOL', 'REGION', 'WRK_ZONE']
feature_cols_I = ['MONTH', 'WKDY_I', 'HOUR_I', 'VEH_INVL', 'NON_INVL', 'LAND_USE', 'MANCOL_I', 'INT_HWY', 'RELJCT_I',
                  'REL_RWY', 'TRAF_WAY', 'NUM_LAN', 'ALIGN_I', 'PROFIL_I', 'SURCON_I', 'TRFCON_I', 'SPDLIM_H', 'LGTCON_I',
                  'WEATHR_I', 'SCHL_BUS', 'PED_ACC', 'ALCHL_I', 'REGION', 'WRK_ZONE']
label = 'CRASH_TYPE'
print("\nDividiendo el conjunto de datos en entrenamiento y test (80%-20%)")
X_train, X_test, y_train, y_test = train_test_data(df, feature_cols, label)
X_I_train, X_I_test, y_I_train, y_I_test = train_test_data(df_I, feature_cols_I, label)

summarize_info(X_train, X_test, y_train, y_test)
summarize_info(X_I_train, X_I_test, y_I_train, y_I_test, "imputados ")

# El clasificador es un árbol de decisión
print("Construyendo el árbol de decisión")
clf = DecisionTreeClassifier()
clf_I = DecisionTreeClassifier()
#clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Entrenando el árbol
print("Entrenando el árbol de decisión")
clf = clf.fit(X_train, y_train)
clf_I = clf_I.fit(X_I_train, y_I_train)

# Prediciendo la etiqueta
print("Prediciendo etiquetas")
y_pred = clf.predict(X_test)
y_I_pred = clf.predict(X_I_test)

# Accuracy del modelo
print("Accuracy: {}".format(accuracy_score(y_test, y_pred)))
print("Accuracy del conjunto imputado: {}".format(accuracy_score(y_I_test, y_I_pred)))

#draw_png(clf, feature_cols)
