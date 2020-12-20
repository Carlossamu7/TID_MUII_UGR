#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Carlos Santiago Sánchez Muñoz
@date: 10 de diciembre de 2020
"""

"""
P4- REGLAS DE ASOCIACIÓN
"""

#############################
#####     LIBRERIAS     #####
#############################

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from apyori import apriori

IMPRIME_INFO = True     # Indica si imprimir información

#############################
#####     FUNCIONES     #####
#############################

""" Lectura de datos. Devuelve el dataframe.
"""
def read_data():
    df = pd.read_excel('prestamo.xls', sheet_name='datos')

    if(IMPRIME_INFO):
        print("\nInformación del dataframe:")
        print(df.info())
        print("\nDataframe:")
        print(df)
    return df

""" Haciendo nominal la variable 'atribute' que contiene 0 y 1. Devuelve el df.
- df: dataframe.
- atribute: nombre del atributo.
"""
def nominalize_0_1(df, atribute):
    for i in range(len(df[atribute])):
        # 0 es que no posee el atributo
        if(df[atribute][i]=="0"):
            df[atribute][i] = "no" + atribute.replace(" ", "")
        # 1 es que sí que posee ese atributo
        else:
            df[atribute][i] = "has" + atribute.replace(" ", "")
    return df

""" Discretizando el atributo 'Education'.
- df: dataframe.
"""
def discretize_education(df):
    for i in range(len(df["Education"])):
        # 1 es Undergraduate
        if(df["Education"][i]=="1"):
            df["Education"][i] = "Undergraduate"
        # 2 es Graduate
        elif(df["Education"][i]=="2"):
            df["Education"][i] = "Graduated"
        # 3 es Advanced/Professional
        else:
            df["Education"][i] = "Advanced/Professional"
    return df

""" Discretizando el atributo 'Family'.
- df: dataframe.
"""
def discretize_family(df):
    for i in range(len(df["Family"])):
        # 1 componente
        if(df["Family"][i]=="1"):
            df["Family"][i] = "1Component"
        # 2 componentes
        elif(df["Family"][i]=="2"):
            df["Family"][i] = "2Component"
        # 3 componentes
        elif(df["Family"][i]=="3"):
            df["Family"][i] = "3Component"
        # 4 componentes
        else:
            df["Family"][i] = "4Component"
    return df

""" Discretizando el atributo 'Age'.
- df: dataframe.
"""
def discretize_age(df):
    for i in range(len(df["Age"])):
        # Twenties
        if(int(df["Age"][i])>=20 and int(df["Age"][i])<30):
            df["Age"][i] = "Twenties"
        # Thirties
        elif(int(df["Age"][i])>=30 and int(df["Age"][i])<40):
            df["Age"][i] = "Thirties"
        # Forties
        elif(int(df["Age"][i])>=40 and int(df["Age"][i])<50):
            df["Age"][i] = "Forties"
        # Fifties
        elif(int(df["Age"][i])>=50 and int(df["Age"][i])<60):
            df["Age"][i] = "Fifties"
        # Sixties
        else:
            df["Age"][i] = "Sixties"
    return df

def inspect(results):
    rh          = [tuple(result[2][0][0]) for result in results]
    lh          = [tuple(result[2][0][1]) for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(rh, lh, supports, confidences, lifts))

########################
#####     MAIN     #####
########################

""" Programa principal. """
def main():
    print("Leyendo el conjunto de datos")
    df = read_data()
    correlaciones = df.corr()
    print(correlaciones)
    print("\nDescarto la variable 'Experience' ya que:")
    print(correlaciones['Age'].sort_values(ascending=False))
    df = df.astype(str)
    df = df.drop(columns = ['ZIP Code'])
    df = df.drop(columns = ['Experience'])
    df = nominalize_0_1(df, "Personal Loan")
    df = nominalize_0_1(df, "Securities Account")
    df = nominalize_0_1(df, "CD Account")
    df = nominalize_0_1(df, "CreditCard")
    df = nominalize_0_1(df, "Online")
    df = nominalize_0_1(df, "CreditCard")
    df = discretize_education(df)
    df = discretize_family(df)
    df = discretize_age(df)
    print(df)

    for x in df.columns:
        print(df.groupby([x]).size())


    """
    records = []
    for i in range(df.shape[0]):
        records.append([str(df.values[i,j]) for j in range(df.shape[1])])
    rules = apriori(records, min_support = 0.006, min_confidence = 0.2, min_lift = 3, min_length = 2)
    print(rules)
    results = list(rules)
    print(results[0])
    print(results[1])
    print(results[2])
    #resultDataFrame=pd.DataFrame(inspect(results), columns=['rhs','lhs','support','confidence','lift'])
    #print(resultDataFrame)
    """

    """
    for item in association_rules:
        # first index of the inner list
        # Contains base item and add item
        pair = item[0]
        items = [x for x in pair]
        print("Rule: " + items[0] + " -> " + items[1])

        #second index of the inner list
        print("Support: " + str(item[1]))

        #third index of the list located at 0th
        #of the third index of the inner list

        print("Confidence: " + str(item[2][0][2]))
        print("Lift: " + str(item[2][0][3]))
        print("=====================================")
    """

if __name__ == "__main__":
	main()
