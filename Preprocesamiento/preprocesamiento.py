#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Carlos Santiago Sánchez Muñoz
@date: 1 de noviembre de 2020
"""

"""
P1- PREPROCESAMIENTO
"""

#############################
#####     LIBRERIAS     #####
#############################

# Para el modelo
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import numpy as np

# Selección de características
from sklearn.feature_selection import RFE

# Silenciar warnings
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

# Para imprimir el árbol
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus

IMPRIME_INFO = True     # Indica si imprimir información
IMPUT_MODE = 2          # 0:mean, 1:mode, 2:delete_instances, 3:delete_attributes, 4:predict
SHOW_CONFUSSION = True	# Indica si se quiere imprimir algunas imágenes
DRAW_TREE = True       # Indica si se debe pintar el árbol

#############################
#####     FUNCIONES     #####
#############################

""" Lectura de datos. Devuelve dos df, uno de ellos con las columnas imputadas.
- mini (op): indica si leer 'accidentes_mini'. Por defecto 'True'.
"""
def read_data(mini=True):
    if(mini):
        df = pd.read_excel(r'accidentes_mini.xls', sheet_name='datos')
    else:
        df = pd.read_excel(r'accidentes.xls', sheet_name='datos')

    df_I = df.copy()
    df = df.drop(columns = ['WKDY_I', 'HOUR_I', 'MANCOL_I', 'RELJCT_I', 'ALIGN_I', 'PROFIL_I', 'SURCON_I',
                            'TRFCON_I', 'SPDLIM_H', 'LGTCON_I', 'WEATHR_I', 'ALCHL_I'])
    df_I = df_I.drop(columns = ['WEEKDAY', 'HOUR', 'MAN_COL', 'REL_JCT', 'ALIGN', 'PROFILE', 'SUR_COND',
                                'TRAF_CON', 'SPD_LIM', 'LGHT_CON', 'WEATHER', 'ALCOHOL'])
    if(IMPRIME_INFO):
        print()
        print(df.info())
        print()
        print(df_I.info())
        #for x in df.columns:
        #    print(df.groupby([x]).size())
        #for x in df_I.columns:
        #    print(df_I.groupby([x]).size())
    return df, df_I

#################################
#####     DISCRETIZANDO     #####
#################################

""" Discretizando la variable 'HOUR'. Devuelve el df.
- df: dataframe.
- imputed (op): si es el imputado. Por defecto 'False'.
"""
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

""" Discretizando la variable 'WEEKDAY'. Devuelve el df.
- df: dataframe.
- imputed (op): si es el imputado. Por defecto 'False'.
"""
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

""" Discretizando la variable 'SPD_LIM'. Devuelve el df.
- df: dataframe.
- imputed (op): si es el imputado. Por defecto 'False'.
"""
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
""" Discretización: 'HOUR', 'WEEKDAY' y 'SPD_LIM'. Devuelve el df.
- df: dataframe.
- df_I: dataframe imputado.
"""
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
    df = discretize_SPD_LIM(df)    # Se elimina esta variable que es desconocida
    df_I = discretize_SPD_LIM(df_I, True)
    if(IMPRIME_INFO):
        print("\n--> Variable velocidad límite")
        print(df.groupby(['SPD_LIM']).size())
        print(df_I.groupby(['SPDLIM_H']).size())

    return df, df_I

##################################################
#####   IMPUTANDO LOS VALORES DESCONOCIDOS   #####
##################################################

""" Imputando valores con la media. Devuelve el df.
- df: dataframe.
"""
def imput_mean(df):
    atr_imputed = ['WEEKDAY', 'HOUR', 'MAN_COL', 'INT_HWY', 'REL_JCT', 'ALIGN', 'PROFILE', 'SUR_COND',
                   'TRAF_CON', 'SPD_LIM', 'LGHT_CON', 'WEATHER', 'PED_ACC', 'ALCOHOL']
    unknown_val = [9, 99, 99, 9, 99, 9, 9, 9, 99, 99, 9, 9, 9999, 9]
    df['PED_ACC'] = df['PED_ACC'].replace(to_replace = 9998, value = 9999)
    for i in range(len(atr_imputed)):
        df[atr_imputed[i]] = df[atr_imputed[i]].replace(to_replace = unknown_val[i], value = None)
        mean = int(df[atr_imputed[i]].mean())
        if(IMPRIME_INFO):
            print("  Cambiando en {} el valor {} por su media: {}".format(atr_imputed[i], unknown_val[i], mean))
        df[atr_imputed[i]] = df[atr_imputed[i]].replace(to_replace = unknown_val[i], value = mean)
    return df

""" Imputando valores con la moda. Devuelve el df.
- df: dataframe.
"""
def imput_mode(df):
    atr_imputed = ['WEEKDAY', 'HOUR', 'MAN_COL', 'INT_HWY', 'REL_JCT', 'ALIGN', 'PROFILE', 'SUR_COND',
                   'TRAF_CON', 'SPD_LIM', 'LGHT_CON', 'WEATHER', 'PED_ACC', 'ALCOHOL']
    unknown_val = [9, 99, 99, 9, 99, 9, 9, 9, 99, 99, 9, 9, 9999, 9]
    df['PED_ACC'] = df['PED_ACC'].replace(to_replace = 9998, value = 9999)
    for i in range(len(atr_imputed)):
        df[atr_imputed[i]] = df[atr_imputed[i]].replace(to_replace = unknown_val[i], value = None)
        mode = int(df[atr_imputed[i]].mode())
        if(IMPRIME_INFO):
            print("  Cambiando en {} el valor {} por su moda: {}".format(atr_imputed[i], unknown_val[i], mode))
        df[atr_imputed[i]] = df[atr_imputed[i]].replace(unknown_val[i], mode)
    return df

""" Borrando intancias con valores desconocidos. Devuelve el df.
- df: dataframe.
"""
def delete_instances(df):
    # No considero SPEED_LIM que siempre es desconocido.
    atr_imputed = ['WEEKDAY', 'HOUR', 'MAN_COL', 'INT_HWY', 'REL_JCT', 'ALIGN', 'PROFILE', 'SUR_COND', 'TRAF_CON',
                   'LGHT_CON', 'WEATHER', 'PED_ACC', 'PED_ACC', 'ALCOHOL']
    unknown_val = [9, 99, 99, 9, 99, 9, 9, 9, 99, 9, 9, 9998, 9999, 9]
    to_delete = []
    for i in range(len(df)):
        cond = False
        for j in range(len(atr_imputed)):
            if(df.iloc(0)[i][atr_imputed[j]]==unknown_val[j]):
                cond = True
                to_delete.append(i)
    if(IMPRIME_INFO):
        print("Número de instancias a eliminar: {}".format(len(to_delete)))
    df = df.drop(to_delete, axis=0)
    df = df.reset_index()
    del df['index']
    return df

""" Borrando características que contienen valores desconocidos. Devuelve el df.
- df: dataframe.
"""
def delete_attributes(df):
    # No considero SPEED_LIM que siempre es desconocido.
    atr_imputed = ['WEEKDAY', 'HOUR', 'MAN_COL', 'INT_HWY', 'REL_JCT', 'ALIGN', 'PROFILE', 'SUR_COND', 'TRAF_CON',
                   'SPD_LIM', 'LGHT_CON', 'WEATHER', 'PED_ACC', 'ALCOHOL']
    if(IMPRIME_INFO):
        print("Número de atributos a eliminar: {}".format(len(atr_imputed)))
    df = df.drop(columns = atr_imputed)
    return df

""" Proceso de imputación. El tipo de imputación lo indica IMPUT_MODE. Devuelve el df.
- df: dataframe.
"""
def imput(df):
    if(IMPUT_MODE==0):
        print("\nImputando algunos valores con la media")
        df = imput_mean(df)
    elif(IMPUT_MODE==1):
        print("\nImputando algunos valores con la moda")
        df = imput_mode(df)
    elif(IMPUT_MODE==2):
        print("\nBorrando instancias con valores desconocidos")
        df = delete_instances(df)
    elif(IMPUT_MODE==3):
        print("\nBorrando atributos con valores desconocidos")
        df = delete_attributes(df)
    else:
        print("\nPrediciento atributos")
        df = predict_values(df)
    return df

############################################
#####   SELECCIÓN DE CARACTERÍSTICAS   #####
############################################

""" Selecciona las características más relevantes usando Recursive Feature Elimination (RFE).
Devuelve el df.
- X: datos de entrada.
- y: etiquetas.
- n_features: número de características.
- feature_cols: nombres de las columnas.
"""
def select_features(X, y, n_features, feature_cols):
    # Selección de características
    selector = RFE(DecisionTreeClassifier(), n_features)
    selector = selector.fit(X, y)
    to_select = selector.support_
    to_delete_attributes = []
    features = []
    for i in range(len(feature_cols)):
        if(not to_select[i]):
            to_delete_attributes.append(feature_cols[i])
        else:
            features.append(feature_cols[i])
    if(IMPRIME_INFO):
        print("\nNum Features: %s" % (selector.n_features_))
        print("Selected Features: %s" % (selector.support_))
        print("Feature Ranking: %s" % (selector.ranking_))
        print(to_delete_attributes)
    return X.drop(columns = to_delete_attributes), features

#######################################
#####   SELECCIÓN DE INSTANCIAS   #####
#######################################

""" Seleccionando intancias con muestreo aleatorio. Devuelve el df.
- df: dataframe.
- frac: fracción entre 0 y 1 de los datos a coger.
"""
def select_instances(df, frac):
    df = df.sample(frac=frac, random_state=1)
    if(IMPRIME_INFO):
        print("  Selección de {} instancias aleatorias".format(df.shape[0]))
    df=df.reset_index()
    del df['index']
    return df

""" Borra una clase para testear el desbalanceo de datos.
- df: dataframe.
- clas: clase a borrar.
"""
def delete_class(df, clas):
    to_delete = []
    for i in range(len(df)):
        if(df.iloc(0)[i]["CRASH_TYPE"]==clas):
            to_delete.append(i)
    if(IMPRIME_INFO):
        print("Número de instancias a eliminar: {}".format(len(to_delete)))
    df = df.drop(to_delete, axis=0)
    df=df.reset_index()
    del df['index']
    return df

""" Elige aleatoriamente tantas instancias de la clase distinta a la clase '2' como instancias tiene '2'.
- df: dataframe.
"""
def select_n_instances(df):
    #seleccionamos un subconjunto
    num = df.groupby(['CRASH_TYPE']).size()[2] # número de instancias de la clase 2
    df_copy = delete_class(df, 2)
    df_copy = df_copy.sample(n=num, random_state=1)
    if(IMPRIME_INFO):
        print("Selección de {} instancias aleatorias".format(df_copy.shape[0]))
    df_copy=df_copy.reset_index()
    del df_copy['index']
    df_2 = df[df["CRASH_TYPE"] == 2]
    df = pd.concat([df_2, df_copy])
    df = df.reset_index()
    del df['index']
    return df

###############################
#####   OTRAS FUNCIONES   #####
###############################

""" Construyendo la variable de clase 'CRASH_TYPE'. Devuelve el df.
- df: dataframe.
"""
def construct_class_variable(df):
    df['CRASH_TYPE'] = df['INJURY_CRASH'] + 2*df['FATALITIES']
    return df.drop(columns = ['PRPTYDMG_CRASH', 'INJURY_CRASH', 'FATALITIES'])

""" Dividiendo en entrada y salida el df. Devuelve las entradas y salidas.
- df: dataframe.
- feature_cols: nombres de las columnas.
- label: etiqueta a predecir.
"""
def X_y_data(df, feature_cols, label):
    return df[feature_cols], df[label]

""" Dividiendo en entrada y salida el df.
- X_train: datos de entrenamiento.
- X_test: datos de test.
- y_train: etiquetas del entrenamiento.
- y_test: etiquetas de test.
- title (op): título. Por defecto la cadena vacía.
"""
def summarize_info(X_train, X_test, y_train, y_test, title=""):
    print("\n------ Información del conjunto de datos " + title + "------")
    print("Tamaño de X_train: {}".format(X_train.shape))
    print("Tamaño de X_test: {}".format(X_test.shape))
    print("Tamaño de y_train: {}".format(y_train.shape))
    print("Tamaño de y_test: {}".format(y_test.shape))

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

""" Dividiendo en entrada y salida el df.
- clf: clasificador del modelo.
- feature_cols: nombres de las columnas.
- title (op): título del grafo. Por defecto 'arbol.png'.
"""
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

""" Dividiendo en entrada y salida el df.
- X_train: datos de entrenamiento.
- X_test: datos de test.
- y_train: etiquetas del entrenamiento.
- y_test: etiquetas de test.
- feature_cols: nombres de las columnas.
- draw_t_title: título del árbol.
- show_conf: indica si se debe de mostrar la matriz de confusión. Por defecto 'True'.
"""
def run_model(X_train, X_test, y_train, y_test, feature_cols, draw_t_title, show_conf=True):
    # El clasificador es un árbol de decisión
    print("Construyendo el árbol de decisión")
    clf = DecisionTreeClassifier(max_depth=10)
    #clf = DecisionTreeClassifier(criterion="entropy", max_depth=10)
    # Entrenando el árbol
    print("Entrenando el árbol de decisión")
    clf = clf.fit(X_train, y_train)
    # Prediciendo la etiqueta
    print("Prediciendo etiquetas")
    y_pred = clf.predict(X_test)
    # Accuracy del modelo
    print("Accuracy: {}".format(accuracy_score(y_test, y_pred)))
    # Matriz de confusión
    if(IMPRIME_INFO):
        print("Matriz de confusión:")
        print(confusion_matrix(y_test, y_pred))
    if(SHOW_CONFUSSION and show_conf):
        show_confussion_matrix(y_test, y_pred, "sin normalizar", False)
        show_confussion_matrix(y_test, y_pred, "normalizada")
    # Pintando el árbol
    if(DRAW_TREE):
        print("Pintando el árbol")
        draw_png(clf, feature_cols, draw_t_title)

""" Todo el proceso de imputación, discretización, selección de atributos/instancias y clasificación
del árbol. Este método es útil para replicarlo luego en el desbalanceo de datos al enfrentar la clase minoritaria
contra el resto.
- df: dataframe.
- df_I: dataframe imputado.
- show_conf: indica si se debe de mostrar la matriz de confusión. Por defecto 'True'.
- draw_t_title: título del árbol. Por defecto "tree".
"""
def process(df, df_I, show_conf=True, draw_t_title="tree"):
    # Conjuntos de entrenamiento y test
    if(IMPUT_MODE==3):
        feature_cols = ['MONTH', 'VEH_INVL', 'NON_INVL', 'LAND_USE', 'REL_RWY', 'TRAF_WAY', 'NUM_LAN',
                        'SCHL_BUS', 'REGION', 'WRK_ZONE']
    else:
        feature_cols = ['MONTH', 'WEEKDAY', 'HOUR', 'VEH_INVL', 'NON_INVL', 'LAND_USE', 'MAN_COL', 'INT_HWY', 'REL_JCT',
                        'REL_RWY', 'TRAF_WAY', 'NUM_LAN', 'ALIGN', 'PROFILE', 'SUR_COND', 'TRAF_CON', 'SPD_LIM', 'LGHT_CON',
                        'WEATHER', 'SCHL_BUS', 'PED_ACC', 'ALCOHOL', 'REGION', 'WRK_ZONE']
    feature_cols_I = ['MONTH', 'WKDY_I', 'HOUR_I', 'VEH_INVL', 'NON_INVL', 'LAND_USE', 'MANCOL_I', 'INT_HWY', 'RELJCT_I',
                      'REL_RWY', 'TRAF_WAY', 'NUM_LAN', 'ALIGN_I', 'PROFIL_I', 'SURCON_I', 'TRFCON_I', 'SPDLIM_H', 'LGTCON_I',
                      'WEATHR_I', 'SCHL_BUS', 'PED_ACC', 'ALCHL_I', 'REGION', 'WRK_ZONE']
    label = 'CRASH_TYPE'
    print("\nDividiendo el conjunto de datos en entrada (X) y salida (y)")
    X, y = X_y_data(df, feature_cols, label)
    X_I, y_I = X_y_data(df_I, feature_cols_I, label)

    print("\nSeleccionando las características más importantes")
    X, feature_cols = select_features(X, y, 15, feature_cols)
    X_I, feature_cols_I = select_features(X_I, y_I, 15, feature_cols_I)

    print("\nDividiendo el conjunto de datos en entrenamiento y test (80%-20%)")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) # 80% training Y 20% test
    X_I_train, X_I_test, y_I_train, y_I_test = train_test_split(X_I, y_I, test_size=0.2, random_state=1)

    summarize_info(X_train, X_test, y_train, y_test)
    summarize_info(X_I_train, X_I_test, y_I_train, y_I_test, "imputados ")

    # EJECUTANDO EL ALGORITMO SOBRE LOS DATOS IMPUTADOS Y SIN IMPUTAR
    print("\n------ Ejecutando el modelo sobre los datos sin imputar ------")
    run_model(X_train, X_test, y_train, y_test, feature_cols, draw_t_title + ".png", show_conf)
    print("\n------ Ejecutando el modelo sobre los datos imputados ------")
    run_model(X_I_train, X_I_test, y_I_train, y_I_test, feature_cols_I, draw_t_title + "_I.png", show_conf)


########################
#####     MAIN     #####
########################

""" Programa principal. """
def main():
    print("Leyendo el conjunto de datos")
    df, df_I = read_data()

    print("\nSeleccionando instancias")
    df = select_instances(df, 0.7)
    df_I = select_instances(df_I, 0.7)

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

    # Imputando valores con la media o moda sobre df
    df = imput(df)

    # Discretización (es preferibles que se haga después de imputar)
    if(IMPUT_MODE!=3):  # En ese caso las variables a discretizar han sido eliminadas
        df, df_I = discretize(df, df_I)

    process(df, df_I)

    # DESBALANCEO DE DATOS: enfrento la clase minoritaria contra el resto.
    print("\n---------------------------------------------------------------------")
    print("------ DESBALANCEO DE DATOS: clase minoritaria contra el resto ------")
    print("---------------------------------------------------------------------")
    print("\n------ Borrada la clase 1 ------\n")
    df_0_2 = delete_class(df, 1)
    df_0_2 = select_n_instances(df_0_2)
    df_0_2_I = delete_class(df_I, 1)
    df_0_2_I = select_n_instances(df_0_2_I)
    process(df_0_2, df_0_2_I, False, "tree_0_2")    # Quedan 8 instancias por lo que queda muy regular

    print("\n------ Borrada la clase 0 ------\n")
    df_1_2 = delete_class(df, 0)
    df_1_2 = select_n_instances(df_1_2)
    df_1_2_I = delete_class(df_I, 0)
    df_1_2_I = select_n_instances(df_1_2_I)
    process(df_1_2, df_1_2_I, False, "tree_1_2")   # Quedan 8 instancias por lo que queda muy regular


if __name__ == "__main__":
	main()
