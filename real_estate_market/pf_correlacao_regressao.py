import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_csv('real_estate_market/data_base/house_info.csv')
Index = (['id', 'date', 'price', 'bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long', 'sqft_living15', 'sqft_lot15'])
X = dataset.iloc[:, [2, 3, 4, 5, 6, 7, 12, 13, 14, 19, 20]]
X = pd.DataFrame(X)
# Apagar algumas colunas que numeros muito parecidos ou nao tem sentido ou nao tem como trabalhar, como id ou data.
dataset.drop(labels=['id', 'date', 'sqft_living', 'sqft_lot'], axis=1, inplace=True)

# IDENTIFICANDO CORRELACAO
dataset.corr()
#                   price  bedrooms  ...  sqft_living15  sqft_lot15
# price          1.000000  0.308350  ...       0.585379    0.082447
# bedrooms       0.308350  1.000000  ...       0.391638    0.029244
# ........
# TABELA DE CORRELAÇÃO:
# 0 a +-0.19 → Correlacao bem fraca
# +-0.20 a +- 0.39 Correlacao fraca
# +-0.40 a +-0.69 Correlacao moderada
# +-0.70 a +-0.89 Correlacao forte
# +-0.90 a +-1.00 Correlacao muito forte

# sns.heatmap(dataset.corr(), annot=True) # No colab o grafico ficou melhor)

from yellowbrick.target import FeatureCorrelation
grafico = FeatureCorrelation(labels=dataset.columns[1:])
grafico.fit(dataset.iloc[:, 1:].values, dataset.iloc[:, 0].values)
# grafico.show() 

# ***************************************************************************************************
# TESTE 01 (T01) - REGRESSAO LINEAR SIMPLES.
dataset = pd.read_csv('real_estate_market/data_base/house_info.csv')
X = dataset['sqft_living'].values
y = dataset['price'].values

X = X.reshape(-1, 1)

from sklearn.model_selection import train_test_split
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size = 0.2, random_state = 1)
X_treinamento.shape, X_teste.shape # R:(17290,) (4323,)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_treinamento, y_treinamento)

# b0
regressor.intercept_
# b1
regressor.coef_

regressor.predict(np.array([[500]]))
# Quantidade de m2, sai o valor da casa

plt.scatter(X, y)
plt.plot(X, regressor.predict(X), color='red')
# plt.show()

# AVALIANDO O MODELO DE REGRESSÃO
regressor.score(X_treinamento, y_treinamento) # 0.48876078213887486
regressor.score(X_teste, y_teste) # 0.5033019006466926
# Quanto maior o score, mais proximo de 1, melhor e mais forte ele é.

# Mean absolute error (MAE)
# Mean squared error (MSE)
# Root mean squared error (RMSE)
''' Objetivo quando se faz analise das métricas é ter o menor valor de error.
Quanto menor o erro, mais proximo as previsoes dos valores reais.'''

previsoes = regressor.predict(X_teste)

from sklearn.metrics import mean_absolute_error, mean_squared_error
mean_absolute_error(y_teste, previsoes) # R: 178578.01
mean_squared_error(y_teste, previsoes) # R: 85730574276.16
math.sqrt(mean_squared_error(y_teste, previsoes)) # R: 292797.83

# ***************************************************************************************************
# TESTE 02 (T02) - REGRESSAO LINEAR MULTIPLA
# escolhe os atributos que tem uma correlação maior, mais q moderada
dataset = pd.read_csv('real_estate_market/data_base/house_info.csv')
dataset.drop(labels=['id', 'date'], axis=1, inplace=True)
X = dataset.iloc[:, [2, 3, 9, 10]].values
y = dataset.iloc[:, 0].values # ou dataset.iloc['price'].values

X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size = 0.2, random_state = 1)

regressor = LinearRegression()
regressor.fit(X_treinamento, y_treinamento)

regressor.score(X_treinamento, y_treinamento) # R: 0.5429158715490658
regressor.score(X_teste, y_teste) # R: 0.5433655885394333

previsoes = regressor.predict(X_teste)
mean_absolute_error(y_teste, previsoes) # R: 163331.28566448076
mean_squared_error(y_teste, previsoes) # R: 78815542841.3
math.sqrt(mean_squared_error(y_teste, previsoes)) # R: 280741.06

# ***************************************************************************************************
# TESTE 03 (T03) - REGRESSAO LINEAR MULTIPLA TRATANDO Y
# Dados com precisão muito fraca, entao deve-se tratar o y, para adequa-lo mais proximo de uma
# distribuicao normal e assim ter melhores resultados
y = np.log(y)

X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size = 0.2, random_state = 1)

regressor = LinearRegression()
regressor.fit(X_treinamento, y_treinamento)

regressor.score(X_treinamento, y_treinamento) # R: 0.559462085496587
regressor.score(X_teste, y_teste) # R: 0.583736323335128

previsoes = regressor.predict(X_teste)
mean_absolute_error(y_teste, previsoes) # R: 0.2763191511945674
mean_squared_error(y_teste, previsoes) # R: 0.12021149927144295
math.sqrt(mean_squared_error(y_teste, previsoes)) # 0.34671530002502476

# ***************************************************************************************************
# TESTE 04 (T04) - REGRESSAO LINEAR MULTIPLA TRATANDO Y
dataset = pd.read_csv('real_estate_market/data_base/house_info.csv')
dataset.drop(labels=['id', 'date'], axis=1, inplace=True)
X = dataset.iloc[:, 1:]
y = dataset.iloc[:, 0].values # ou dataset.iloc['price'].values

y = np.log(y)

X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size = 0.2, random_state = 1)

regressor = LinearRegression()
regressor.fit(X_treinamento, y_treinamento)

regressor.score(X_treinamento, y_treinamento) # R: 0.7688599782083718
regressor.score(X_teste, y_teste) # R: 0.7758399045656308

previsoes = regressor.predict(X_teste)
mean_absolute_error(y_teste, previsoes) # R: 0.1948395068561904
mean_squared_error(y_teste, previsoes) # R: 0.06473450041303892
math.sqrt(mean_squared_error(y_teste, previsoes)) # 0.2544297553609619

# ***************************************************************************************************
# TESTE 05 (T05) - REGRESSAO LINEAR SELECIONANDO OS ATRIBUTOS E TRATANDO Y
dataset = pd.read_csv('real_estate_market/data_base/house_info.csv')
dataset.drop(labels=['id', 'date'], axis=1, inplace=True)

from sklearn.feature_selection import SelectFdr, f_regression
selecao = SelectFdr(f_regression, alpha=0.00)
X_novo = selecao.fit_transform(X, y)
X.shape, X_novo.shape # R: (21613, 18) (21613, 10)

y = dataset.iloc[:, 0].values # ou dataset.iloc['price'].values
y = np.log(y)

X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X_novo, y, test_size = 0.2, random_state = 1)

regressor = LinearRegression()
regressor.fit(X_treinamento, y_treinamento)

regressor.score(X_treinamento, y_treinamento) # R: 0.7260052758762527
regressor.score(X_teste, y_teste) # R: 0.7400466783307362

previsoes = regressor.predict(X_teste)
mean_absolute_error(y_teste, previsoes) # R: 0.20948193987525873
mean_squared_error(y_teste, previsoes) # R: 0.07507111547379215
math.sqrt(mean_squared_error(y_teste, previsoes)) # 0.2739910864860245

# ***************************************************************************************************
# TESTE 06 (T06) - CATBOOST
dataset = pd.read_csv('real_estate_market/data_base/house_info.csv')
dataset.drop(labels=['id', 'date'], axis=1, inplace=True)

X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values # ou dataset.iloc['price'].values

import catboost as ctb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics

X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size = 0.2, random_state = 1)
model_cat_class = ctb.CatBoostRegressor()
model_cat_class.fit(X_treinamento, y_treinamento)
prev_catboost = model_cat_class.predict(X_teste)
acerto_cat_t6 = metrics.r2_score(y_teste, prev_catboost)

# ***************************************************************************************************
# TESTE 07 (T07) - CATBOOST SELECIONANDO ATRIBUTOS PREVISORES

from sklearn.feature_selection import SelectFdr, f_regression
selecao = SelectFdr(f_regression, alpha=0.00)
X_novo = selecao.fit_transform(X, y)

X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X_novo, y, test_size = 0.2, random_state = 1)
model_cat_class = ctb.CatBoostRegressor()
model_cat_class.fit(X_treinamento, y_treinamento)
prev_catboost = model_cat_class.predict(X_teste)
acerto_cat_t7 = metrics.r2_score(y_teste, prev_catboost)

# ***************************************************************************************************
# TESTE 08 (T08) - CATBOOST TRATANDO Y

y = np.log(y)
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size = 0.2, random_state = 1)
model_cat_class = ctb.CatBoostRegressor()
model_cat_class.fit(X_treinamento, y_treinamento)
prev_catboost = model_cat_class.predict(X_teste)
acerto_cat_t8 = metrics.r2_score(y_teste, prev_catboost)

# ***************************************************************************************************
# TESTE 09 (T09) - CATBOOST SELECIONANDO ATRIBUTOS PREVISORES E TRATANDO Y

X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X_novo, y, test_size = 0.2, random_state = 1)
model_cat_class = ctb.CatBoostRegressor()
model_cat_class.fit(X_treinamento, y_treinamento)
prev_catboost = model_cat_class.predict(X_teste)
acerto_cat_t9 = metrics.r2_score(y_teste, prev_catboost)

print(f'Cat-T06:{acerto_cat_t6}') # R: 0.8868034842592303
print(f'Cat-T07:{acerto_cat_t7}') # R: 0.8149249719289815
print(f'Cat-T08:{acerto_cat_t8}') # R: 0.9128171224461752
print(f'Cat-T09:{acerto_cat_t9}') # R: 0.8534832252055329

"""
Dos testes realizados o que teve melhor performance foi utilizando Catboost, 
com todos atributos previsores e tratamento do atributo alvo (y).
"""

