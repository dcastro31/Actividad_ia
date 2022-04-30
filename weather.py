# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 21:33:05 2022

"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

url = 'weatherAUS.csv'
data = pd.read_csv(url)

# Tratamiento de data

rangos = [-8.0,0,10,20,30,40]
nombres = ['1','2','3','4','5']
data.MinTemp = pd.cut(data.MinTemp, rangos, labels=nombres)


rangosMax = [0,10,20,30,40,50]
nombresMax = ['1','2','3','4','5']
data.MaxTemp = pd.cut(data.MaxTemp, rangosMax, labels=nombresMax)


rangosRai = [-1,50,100,150,200,250]
nombresRai = ['1','2','3','4','5']
data.Rainfall = pd.cut(data.Rainfall, rangosRai, labels=nombresRai)


rangosEva = [-1,20,40,60,80,100]
nombresEva = ['1','2','3','4','5']
data.Evaporation = pd.cut(data.Evaporation, rangosEva, labels=nombresEva)


rangosSun = [-1,5,10,15]
nombresSun = ['1','2','3']
data.Sunshine = pd.cut(data.Sunshine, rangosSun, labels=nombresSun)



rangosWinG = [0,30,60,90,130]
nombresWinG = ['1','2','3','4']
data.WindGustSpeed = pd.cut(data.WindGustSpeed, rangosWinG, labels=nombresWinG)


rangosWinS9 = [0,20,40,60,80]
nombresWinS9 = ['1','2','3','4']
data.WindSpeed9am = pd.cut(data.WindSpeed9am, rangosWinS9, labels=nombresWinS9)


rangosWinS3 = [0,20,40,60,80]
nombresWinS3 = ['1','2','3','4']
data.WindSpeed3pm = pd.cut(data.WindSpeed3pm, rangosWinS3, labels=nombresWinS3)


rangosHum9 = [-1,20,40,60,80,101]
nombresHum9 = ['1','2','3','4','5']
data.Humidity9am = pd.cut(data.Humidity9am, rangosHum9, labels=nombresHum9)


rangosHum3 = [-1,20,40,60,80,101]
nombresHum3 = ['1','2','3','4','5']
data.Humidity3pm = pd.cut(data.Humidity3pm, rangosHum3, labels=nombresHum3)


rangosPre9 = [980,1000,1050]
nombresPre9 = ['1','2']
data.Pressure9am = pd.cut(data.Pressure9am, rangosPre9, labels=nombresPre9)


rangosPre3 = [970,1000,1040]
nombresPre3 = ['1','2']
data.Pressure3pm = pd.cut(data.Pressure3pm, rangosPre3, labels=nombresPre3)


rangosTem9 = [-0.5,0,20,40]
nombresTem9 = ['1','2','3']
data.Temp9am = pd.cut(data.Temp9am, rangosTem9, labels=nombresTem9)


rangosTem3 = [0,20,40,50]
nombresTem3 = ['1','2','3']
data.Temp3pm = pd.cut(data.Temp3pm, rangosTem3, labels=nombresTem3)


data['RainToday'].replace(['No', 'Yes'], [0, 1], inplace=True)
data['RainTomorrow'].replace(['No', 'Yes'], [0, 1], inplace=True)


data.dropna(axis=0,how='any', inplace=True)

data.drop(['Date','Location','WindGustDir','WindDir9am','WindDir3pm','RISK_MM'], axis= 1, inplace = True)

# partir la data en dos

data_train = data[:40000]
data_test = data[40000:]


x = np.array(data_train.drop(['RainTomorrow'], 1))
y = np.array(data_train.RainTomorrow)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_test_out = np.array(data_test.drop(['RainTomorrow'], 1))
y_test_out = np.array(data_test.RainTomorrow)


#REGRESION LOGISTICA 

# Modelo
logreg = LogisticRegression(solver='lbfgs', max_iter = 7600)

#EEntreno el modelo
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
