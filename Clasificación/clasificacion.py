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

########################
#####     MAIN     #####
########################

""" Programa principal. """
def main():
    print("Leyendo el conjunto de datos")
    df = read_data()

    X, y = split_data(df)
    

if __name__ == "__main__":
	main()
