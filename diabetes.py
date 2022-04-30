# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 20:40:52 2022

"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

url = 'diabetes.csv'
data = pd.read_csv(url)

# Tratamiento de data
rangos = [0,6,12,17]
nombres = ['1','2','3']
data.Pregnancies = pd.cut(data.Pregnancies, rangos, labels=nombres)
data.Pregnancies.replace(np.nan, 1, inplace=True)


rangosGlu = [0,50,100,150,200]
nombresGlu = ['1','2','3','4']
data.Glucose = pd.cut(data.Glucose, rangosGlu, labels=nombresGlu)

rangosBLo = [0,50,100,150]
nombresBlo = ['1','2','3']
data.BloodPressure = pd.cut(data.BloodPressure, rangosBlo, labels=nombresBlo)

rangosIns = [-1,200,500,700,900]
nombresIns = ['1','2','3','4']
data.Insulin = pd.cut(data.Insulin, rangosIns, labels=nombresIns)

rangosBMI = [-1,20,40,60]
nombresBMI = ['1','2','3']
data.BMI = pd.cut(data.BMI, rangosBMI, labels=nombresBMI)


rangosAge = [20,40,60,90]
nombresAge = ['1','2','3']
data.Age = pd.cut(data.Age, rangosAge, labels=nombresAge)

data.dropna(axis=0,how='any', inplace=True)


#Columnas Innecesarias
data.drop(['SkinThickness','DiabetesPedigreeFunction'], axis= 1, inplace = True)

# Partir la data en dos

data_train = data[:450]
data_test = data[450:]

x = np.array(data_train.drop(['Outcome'], 1))
y = np.array(data_train.Outcome) 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_test_out = np.array(data_test.drop(['Outcome'], 1))
y_test_out = np.array(data_test.Outcome) 


#REGRESION LOGISTICA 

# Modelo
logreg = LogisticRegression(solver='lbfgs', max_iter = 7600)

# Entreno el modelo
logreg.fit(x_train,y_train)

# METRICAS

print('*'*50)
print('Regresión Logística')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {logreg.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {logreg.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {logreg.score(x_test_out, y_test_out)}')


# MAQUINA DE SOPORTE VECTORIAL

# Modelo
svc = SVC(gamma='auto')

# Entreno el modelo
svc.fit(x_train, y_train)

# MÉTRICAS

print('*'*50)
print('Maquina de soporte vectorial')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {svc.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {svc.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {svc.score(x_test_out, y_test_out)}')


# ARBOL DE DECISIÓN

# Modelo
arbol = DecisionTreeClassifier()

# Entreno el modelo
arbol.fit(x_train, y_train)

# MÉTRICAS

print('*'*50)
print('Decisión Tree')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {arbol.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {arbol.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {arbol.score(x_test_out, y_test_out)}')


# RANDOM FOREST

# Modelo
forest = RandomForestClassifier()

# Entreno el modelo
forest.fit(x_train, y_train)

# MÉTRICAS

print('*'*50)
print('RANDOM FOREST')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {forest.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {forest.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {forest.score(x_test_out, y_test_out)}')


# NAIVE BAYES

# Modelo
nayve = GaussianNB()

# Entreno el modelo
nayve.fit(x_train, y_train)

# MÉTRICAS

print('*'*50)
print('NAYVE BAYES')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {nayve.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {nayve.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {nayve.score(x_test_out, y_test_out)}')