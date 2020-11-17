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

IMPRIME_INFO = True

def read_data(mini=True):
    if(mini):
        df = pd.read_excel(r'accidentes_mini.xls', sheet_name='datos')
    else:
        df = pd.read_excel(r'accidentes.xls', sheet_name='datos')

    df_I = df.copy()
    df = df.drop(columns = ['WKDY_I', 'HOUR_I', 'MANCOL_I', 'RELJCT_I', 'ALIGN_I', 'PROFIL_I', 'SURCON_I', 'TRFCON_I', 'SPDLIM_H', 'LGTCON_I', 'WEATHR_I', 'ALCHL_I'])
    df_I = df_I.drop(columns = ['WEEKDAY', 'HOUR', 'MAN_COL', 'REL_JCT', 'ALIGN', 'PROFILE', 'SUR_COND', 'TRAF_CON', 'SPD_LIM', 'LGHT_CON', 'WEATHER', 'ALCOHOL'])
    return df, df_I

def discretize_HOUR(df, imputed=False):
    if imputed:
        atribute = 'HOUR_I'
    else:
        atribute = 'HOUR'
    for i in range(len(df[atribute])):
        # Si es por la mañana asigno un 0
        if(df[atribute][i]>=7 and df[atribute][i]<=14):
            df[atribute][i] = 0
        # Si es por la tarde asigno un 1
        elif(df[atribute][i]>=15 and df[atribute][i]<=23):
            df[atribute][i] = 1
        # Si es por la noche o es desconocido asigno un 2
        else:
            df[atribute][i] = 2
    return df

def discretize_WEEKDAY(df, imputed=False):
    if imputed:
        atribute = 'WKDY_I'
    else:
        atribute = 'WEEKDAY'
    for i in range(len(df[atribute])):
        # Si es entre semana asigno 0
        if(df[atribute][i]>=2 and df[atribute][i]<=6):
            df[atribute][i] = 0
        # Si es entre semana asigno 1
        else:
            df[atribute][i] = 1
    return df

def discretize_SPD_LIM(df, imputed=False):
    if imputed:
        atribute = 'SPDLIM_H'
    else:
        atribute = 'SPD_LIM'
    for i in range(len(df[atribute])):
        # Si es menor a 40 asigno 0
        if(df[atribute][i]<40):
            df[atribute][i] = 0
        # Si entre 40 y 65 asigno 1
        elif(df[atribute][i]>=40 and df[atribute][i]<66):
            df[atribute][i] = 1
        # Si es mayor a 65 asigno 2
        else:
            df[atribute][i] = 2
    return df

# A mi entender solo merece la pena discretizar las variables HOUR, WEEKDAY y SPD_LIM
def discretize(df, df_I):
    print("\nDiscretizando algunas variables")

    # Discretizando el atributo HOUR y HOUR_I
    df = discretize_HOUR(df)
    df_I = discretize_HOUR(df_I, True)
    if(IMPRIME_INFO):
        print("\n--> Variable hora")
        print(df.groupby(['HOUR']).size())
        print(df_I.groupby(['HOUR_I']).size())

    # Discretizando el atributo WEEKDAY y WKDY_I
    df = discretize_WEEKDAY(df)
    df_I = discretize_WEEKDAY(df_I, True)
    if(IMPRIME_INFO):
        print("\n--> Variable día de la semana")
        print(df.groupby(['WEEKDAY']).size())
        print(df_I.groupby(['WKDY_I']).size())

    # Discretizando el atributo SPD_LIM y SPDLIM_H
    df = discretize_SPD_LIM(df)
    df_I = discretize_SPD_LIM(df_I, True)
    if(IMPRIME_INFO):
        print("\n--> Variable velocidad límite")
        print(df.groupby(['SPD_LIM']).size())
        print(df_I.groupby(['SPDLIM_H']).size())

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

def run_model(X_train, X_test, y_train, y_test):
    # El clasificador es un árbol de decisión
    print("Construyendo el árbol de decisión")
    clf = DecisionTreeClassifier()
    #clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
    # Entrenando el árbol
    print("Entrenando el árbol de decisión")
    clf = clf.fit(X_train, y_train)
    # Prediciendo la etiqueta
    print("Prediciendo etiquetas")
    y_pred = clf.predict(X_test)
    # Accuracy del modelo
    print("Accuracy: {}".format(accuracy_score(y_test, y_pred)))
    # Pintando el árbol
    #draw_png(clf, feature_cols)

def imput_mean(df):
    atr_imputed = ['WEEKDAY', 'HOUR', 'MAN_COL', 'INT_HWY', 'REL_JCT', 'ALIGN', 'PROFILE', 'SUR_COND', 'TRAF_CON', 'SPD_LIM', 'LGHT_CON', 'WEATHER', 'PED_ACC', 'PED_ACC', 'ALCOHOL']
    unknown_val = [9, 99, 99, 9, 99, 9, 9, 9, 99, 99, 9, 9, 9998, 9999, 9]
    for i in range(len(atr_imputed)):
        mean = int(df[atr_imputed[i]].mean())
        if(IMPRIME_INFO):
            print("Cambiando en {} el valor {} por su media: {}".format(atr_imputed[i], unknown_val[i], mean))
        df[atr_imputed[i]] = df[atr_imputed[i]].replace(to_replace = unknown_val[i], value = mean)
    return df

def imput_mode(df):
    atr_imputed = ['WEEKDAY', 'HOUR', 'MAN_COL', 'INT_HWY', 'REL_JCT', 'ALIGN', 'PROFILE', 'SUR_COND', 'TRAF_CON', 'SPD_LIM', 'LGHT_CON', 'WEATHER', 'PED_ACC', 'PED_ACC', 'ALCOHOL']
    unknown_val = [9, 99, 99, 9, 99, 9, 9, 9, 99, 99, 9, 9, 9998, 9999, 9]
    for i in range(len(atr_imputed)):
        mode = int(df[atr_imputed[i]].mode())
        if(IMPRIME_INFO):
            print("Cambiando en {} el valor {} por su moda: {}".format(atr_imputed[i], unknown_val[i], mode))
        df[atr_imputed[i]] = df[atr_imputed[i]].replace(unknown_val[i], mode)
    return df

########################
#####     MAIN     #####
########################

""" Programa principal. """
def main():
    print("Leyendo el conjunto de datos")
    df, df_I = read_data()

    if(IMPRIME_INFO):
        print(df.info())
        print(df_I.info())
        #for x in df.columns:
        #    print(df.groupby([x]).size())
        #for x in df_I.columns:
        #    print(df_I.groupby([x]).size())

    #df, df_I = discretize(df, df_I)

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

    for x in df.columns:
        print(df.groupby([x]).size())
    print("\n\n\n Escribo \n\n\n")
    df = imput_mean(df)
    print("\n\n\n Borro \n\n\n")
    for x in df.columns:
        print(df.groupby([x]).size())
    #df = imput_mode(df)

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

    # EJECUTANDO EL ALGORITMO SOBRE LOS DATOS IMPUTADOS Y SIN IMPUTAR
    print("\n------ Ejecutando el modelo sobre los datos sin imputar ------")
    run_model(X_train, X_test, y_train, y_test)
    print("\n------ Ejecutando el modelo sobre los datos imputados ------")
    run_model(X_I_train, X_I_test, y_I_train, y_I_test)


if __name__ == "__main__":
	main()
