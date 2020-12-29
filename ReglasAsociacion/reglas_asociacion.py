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
#from apyori import apriori
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules

IMPRIME_INFO = True     # Indica si imprimir información
SUPP = 0.15             # Umbral de soporte
CONF = 0.99             # Umbral de confidence

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

####################################
#####     Preprocesamiento     #####
####################################

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

""" Discretizando el atributo 'Education'. Devuelve el df.
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

""" Discretizando el atributo 'Family'. Devuelve el df.
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

""" Discretizando el atributo 'Age'. Devuelve el df.
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

""" Discretizando el atributo 'Mortgage'. Devuelve el df.
- df: dataframe.
"""
def discretize_mortgage(df):
    for i in range(len(df["Mortgage"])):
        # NoMortgage
        if(df["Mortgage"][i]=="0"):
            df["Mortgage"][i] = "NoMortgage"
        # VeryLowMortgage
        elif(int(df["Mortgage"][i])>0 and int(df["Mortgage"][i])<=100):
            df["Mortgage"][i] = "VeryLowMortgage"
        # LowMortgage
        elif(int(df["Mortgage"][i])>100 and int(df["Mortgage"][i])<=200):
            df["Mortgage"][i] = "LowMortgage"
        # MediumMortgage
        elif(int(df["Mortgage"][i])>200 and int(df["Mortgage"][i])<=300):
            df["Mortgage"][i] = "MediumMortgage"
        # HighMortgage
        elif(int(df["Mortgage"][i])>300 and int(df["Mortgage"][i])<=400):
            df["Mortgage"][i] = "HighMortgage"
        # VeryHighMortgage
        else:
            df["Mortgage"][i] = "VeryHighMortgage"
    return df

""" Discretizando el atributo 'Mortgage'. Devuelve el df.
- df: dataframe.
"""
def discretize_CCAvg(df):
    for i in range(len(df["CCAvg"])):
        # VeryLowCCAvg
        if(float(df["CCAvg"][i])>=0 and float(df["CCAvg"][i])<0.75):
            df["CCAvg"][i] = "VeyLowCCAvg"
        # LowCCAvg
        elif(float(df["CCAvg"][i])>=0.75 and float(df["CCAvg"][i])<1.5):
            df["CCAvg"][i] = "LowCCAvg"
        # HighCCAvg
        elif(float(df["CCAvg"][i])>=1.5 and float(df["CCAvg"][i])<2.5):
            df["CCAvg"][i] = "HighCCAvg"
        # VeryHighCCAvg
        else:
            df["CCAvg"][i] = "VeryHighCCAvg"
    return df

""" Discretizando el atributo 'Mortgage'. Devuelve el df.
- df: dataframe.
"""
def discretize_Income(df):
    for i in range(len(df["Income"])):
        # VerLowIncome
        if(int(df["Income"][i])>=0 and int(df["Income"][i])<40):
            df["Income"][i] = "VeryLowIncome"
        # LowIncome
        elif(int(df["Income"][i])>=40 and int(df["Income"][i])<65):
                df["Income"][i] = "LowIncome"
        # HighIncome
        elif(int(df["Income"][i])>=65 and int(df["Income"][i])<100):
                df["Income"][i] = "HighIncome"
        # VeryHighIncome
        else:
            df["Income"][i] = "VeryHighIncome"
    return df

""" Todo el preprocesado. Devuelve el df preprocesado.
- df: dataframe.
"""
def preprocesamiento(df):
    correlaciones = df.corr()
    df = df.astype(str)
    print("\n---  PREPROCESAMIENTO  ---")
    print("Descarto la variable 'Experience' ya que tiene una correlación de {}".format(correlaciones['Age'].sort_values(ascending=False)[1]))
    df = df.drop(columns = ['Experience'])
    df = df.drop(columns = ['ZIP Code'])
    print("Nominalizando 'Personal Loan'")
    df = nominalize_0_1(df, "Personal Loan")
    print("Nominalizando 'Securities Account'")
    df = nominalize_0_1(df, "Securities Account")
    print("Nominalizando 'CD Account'")
    df = nominalize_0_1(df, "CD Account")
    print("Nominalizando 'CreditCard'")
    df = nominalize_0_1(df, "CreditCard")
    print("Nominalizando 'Online'")
    df = nominalize_0_1(df, "Online")
    print("Discretizando 'Education'")
    df = discretize_education(df)
    print("Discretizando 'Family'")
    df = discretize_family(df)
    print("Discretizando 'Age'")
    df = discretize_age(df)
    print("Discretizando 'Mortgage'")
    df = discretize_mortgage(df)
    print("Discretizando 'CCAvg'")
    df = discretize_CCAvg(df)
    print("Discretizando 'Income'")
    df = discretize_Income(df)
    print()
    print(df)
    return df

######################################
#####    Reglas de asociación    #####
######################################

""" Reglas de asociación. Devuelve las reglas.
- df: dataframe.
"""
def get_rules(df):
    print("\n---  REGLAS DE ASOCIACIÓN  ---")
    records = []
    for i in range(df.shape[0]):
        records.append([str(df.values[i,j]) for j in range(df.shape[1])])

    te = TransactionEncoder()
    te_ary = te.fit(records).transform(records)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    """
    df = df.drop(columns = ['noPersonalLoan'])
    df = df.drop(columns = ['noSecuritiesAccount'])
    df = df.drop(columns = ['noCDAccount'])
    df = df.drop(columns = ['noCreditCard'])
    df = df.drop(columns = ['noOnline'])
    """

    print("\nDataframe preparado para extraer las reglas de asociación:")
    print(df)
    #frequent_itemsets = fpgrowth(df, min_support=0.6, use_colnames=True)
    frequent_itemsets = apriori(df, min_support=SUPP, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=CONF)
    rules = pd.DataFrame(rules)

    to_delete = []
    for i in range(rules.shape[0]):
        if(len(rules["antecedents"][i])>3 or len(rules["consequents"][i])>3):
            to_delete.append(i)

    print("Índices de las reglas que tienen más de 3 antecedentes o consecuentes: ")
    print(to_delete)
    rules = rules.drop(to_delete, axis=0)
    rules = rules.reset_index()
    del rules['index']

    return rules

########################
#####     MAIN     #####
########################

""" Programa principal. """
def main():
    print("Leyendo el conjunto de datos")
    df = read_data()
    print(df["Income"].describe())

    # Preprocesamiento
    df = preprocesamiento(df)

    for x in df.columns:
        print(df.groupby([x]).size())

    # Extrayendo las reglas de asociación
    rules = get_rules(df)

    # Imprimo las reglas
    print("\nREGLAS DE ASOCIACIÓN (support>{}, confidence>{}):".format(SUPP, CONF))
    print(rules)
    # Guardo las reglas en un csv
    print("\nGuardando reglas en fichero 'Reglas.csv'...")
    rules.to_csv('Reglas.csv', header = True, index = False)


if __name__ == "__main__":
	main()
