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
from sklearn import tree

df = pd.read_excel(r'accidentes_mini.xls', sheet_name='datos')
#df = pd.read_excel(r'accidentes.xls', sheet_name='datos')
print(df)

"""
tree_model = tree.DecisionTreeClassifier(criterion='entropy',
                                         min_samples_split=20,
                                         min_samples_leaf=5,
                                         max_depth = depth,
                                         class_weight={1:3.5})

model = tree_model.fit(X = f_train.drop(['top'], axis=1), y = f_train["top"])
valid_acc = model.score(X = f_valid.drop(['top'], axis=1), y = f_valid["top"])
"""
