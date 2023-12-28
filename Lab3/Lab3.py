#!/usr/bin/env python
# coding: utf-8

# ## Лабораторная работа № 3
# ### Андрюшина Мария, 932001
# ### Аренда велосипедов

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# ### Работа с данными

# In[3]:


df = pd.read_csv('hour.csv', index_col = 0)
df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.describe(include=[object])


# Пропущенных значений нет

# In[7]:


df.isnull().sum()


# Извлечём день из признака dteday:

# In[8]:


df['dteday'] = pd.to_datetime(df['dteday'])
df['day'] = df['dteday'].dt.day
df = df.drop(['dteday'], axis=1)
df


# In[10]:


df.columns


# Возьмём для обучения все признаки, кроме casual и registered, так как они являются производными от таргета и не входят в основной набор данных:

# In[46]:


features = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday',
       'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'cnt', 'day']


# In[47]:


df1 = df.copy()
df1 = df1[features]
df1


# Обработаем данные в столбцах season, hr и weathersit:

# In[48]:


df1 = pd.get_dummies(df1, columns = ['season', 'hr', 'weathersit'], dtype=float)
df1.head()


# In[49]:


X = df1.drop(['cnt'],axis=1)
y = df1['cnt']


# In[50]:


min_max_scaler = MinMaxScaler()
X_scaled = min_max_scaler.fit_transform(X)


# Разделим датасет на две части: тренировочную (70%) и тестовую (30%):

# In[51]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=77)


# ### Linear Regression

# По умолчанию поставим n_jobs = -1 для распараллеливания работы. Подберем параметры fit_intercept, copy_X и positive:

# In[85]:


parameters = {'fit_intercept': [True, False], 'copy_X': [True, False], 'positive':  [True, False]}
model = LinearRegression(n_jobs = -1)
gridForest = GridSearchCV(estimator=model, param_grid=parameters)
gridForest.fit(X_train, y_train)
print('Best score:', round(gridForest.best_score_,3))
print('Best model:', gridForest.best_estimator_)


# Оптимальные значения параметров: fit_intercept = True, copy_X = True, positive = False. Эти значения стоят по умолчанию.

# Построим модель LinearRegression с учетом подобранных параметров:

# In[86]:


linear_model = LinearRegression(n_jobs = -1)
linear_model.fit(X_train,y_train)


# Выводим метрики:

# In[87]:


preds = linear_model.predict(X_test)
print('Linear Regression Estimation:')
print('R2-score:',round(r2_score(y_test,preds),3))
print('MSE:',round(mean_squared_error(y_test,preds),3))
print('MAE:',round(mean_absolute_error(y_test,preds),3))


# Получим значимость признаков:

# In[88]:


print('weight_0 =',np.round(linear_model.intercept_,3))
print('weights = ',np.round(linear_model.coef_,3))


# ### Вывод

# При создании модели LinearRegression использовались следующие гиперпараметры: n_jobs = -1. На тестовой выборке мы получили R2-score: 0.669, MSE: 10725.116, MAE: 75.819.

# ### Polynomial Features

# Для PolynomialFeatures возьмём degree, равное 2, так как модель обучается на 42 признаках и при большем значении degree может не хватить вычислительных мощностей.

# In[57]:


quadratic = PolynomialFeatures(degree = 2)
X_train_quadratic = quadratic.fit_transform(X_train)
X_test_quadratic = quadratic.transform(X_test)


# In[58]:


print(X_train.shape)
print(X_train_quadratic.shape)


# Подберем параметры fit_intercept, copy_X и positive:

# In[61]:


parameters = {'fit_intercept': [True, False], 'copy_X': [True, False], 'positive':  [True, False]}
model = LinearRegression(n_jobs = -1)
gridForest = GridSearchCV(estimator=model, param_grid=parameters)
gridForest.fit(X_train_quadratic, y_train)
print('Best score:', round(gridForest.best_score_,3))
print('Best model:', gridForest.best_estimator_)


# Оптимальные значения параметров: fit_intercept = False, copy_X = True, positive = True.

# Построим модель LinearRegression с учетом подобранных параметров:

# In[62]:


quadratic_model = LinearRegression(fit_intercept=False, n_jobs=-1, positive=True)
quadratic_model.fit(X_train_quadratic,y_train)


# Выводим метрики:

# In[63]:


preds = quadratic_model.predict(X_test_cube)
print('QuadraticCubic Regression Estimation:')
print('R2-score:',round(r2_score(y_test,preds),3))
print('MSE:',round(mean_squared_error(y_test,preds),3))
print('MAE:',round(mean_absolute_error(y_test,preds),3))


# Получим значимость признаков:

# In[64]:


print('weight_0 =', np.round(quadratic_model.intercept_, 3))
print('weights = ', np.round(quadratic_model.coef_, 3))


# ### Вывод

# При создании модели LinearRegression с PolynomialFeatures использовались следующие гиперпараметры: fit_intercept=False, n_jobs=-1, positive=True. На тестовой выборке мы получили R2-score: 0.813, MSE: 6067.548, MAE: 53.043.

# ### Random Forest

# In[65]:


X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3,random_state=77)


# Подберем параметры n_estimators, max_features и max_depth:

# In[68]:


parameters = {'n_estimators': [200, 300, 400],
              'max_features': [5, 15, 30, 40], 'max_depth': [3, 7, 9, 12]}
model = RandomForestRegressor(n_jobs = -1, random_state = 77)
gridForest = GridSearchCV(estimator=model, param_grid=parameters)
gridForest.fit(X_train, y_train)
print('Best score:', round(gridForest.best_score_,3))
print('Best model:', gridForest.best_estimator_)


# Оптимальные значения параметров: n_estimators = 300, max_features = 30, max_depth = 12.

# Построим модель RandomForestRegressor с учетом подобранных параметров:

# In[69]:


forest_model = RandomForestRegressor(max_depth=12, max_features=30, n_estimators=300, n_jobs=-1, random_state=77)
forest_model.fit(X_train, y_train)


# Выводим метрики:

# In[70]:


preds = forest_model.predict(X_test)
print('Random Forest Regression Estimation:')
print('R2-score:',round(r2_score(y_test,preds),3))
print('MSE:',round(mean_squared_error(y_test,preds),3))
print('MAE:',round(mean_absolute_error(y_test,preds),3))


# Получим значимость признаков:

# In[71]:


weights = pd.DataFrame({'column': X_train.columns,
                        'weight': forest_model.feature_importances_})
weights.sort_values(by='weight', ascending=False).head()


# In[72]:


plt.figure(figsize=(15,12))
plt.barh(y=weights['column'], width=weights['weight'])


# Визуализируем несколько деревьев решений:

# In[73]:


tree_0 = forest_model.estimators_[0]


# In[74]:


plt.figure(figsize=(25,20))
tree.plot_tree(tree_0, feature_names=X_train.columns, max_depth = 2, filled = True)

plt.show()


# In[75]:


tree_1 = forest_model.estimators_[1]


# In[76]:


plt.figure(figsize=(25,20))
tree.plot_tree(tree_1, feature_names=X_train.columns, max_depth = 3, filled = True)

plt.show()


# ### Вывод

# При создании модели RandomForestRegressor использовались следующие гиперпараметры: max_depth=12, max_features=30, n_estimators=300, n_jobs=-1, random_state=77. На тестовой выборке мы получили R2-score: 0.831, MSE: 5475.327, MAE: 53.644.
# 
# При анализе значимости признаков мы выяснили, что наибольшее влияние на предсказание таргета оказывают следующие признаки: atemp, hr_17, hr_18, hum, yr.	

# ### Gradient Boosting

# In[77]:


min_max_scaler = MinMaxScaler()
X_scaled = min_max_scaler.fit_transform(X)


# In[78]:


X_train,X_test,y_train,y_test = train_test_split(X_scaled, y, test_size=0.3,random_state=77)


# Подберем параметры loss и learning_rate:

# In[79]:


parameters = {'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],
              'learning_rate': [0.01, 0.1, 0.5, 1]}
model = GradientBoostingRegressor(random_state = 77)
gridForest = GridSearchCV(estimator=model, param_grid=parameters)
gridForest.fit(X_train, y_train)
print('Best score:', round(gridForest.best_score_,3))
print('Best model:', gridForest.best_estimator_)


# Оптимальные значения параметра learning_rate = 0.5.

# Подберем параметры n_estimators, max_features и max_depth:

# In[80]:


parameters = {'n_estimators': [200, 300, 400],
              'max_features': [5, 15, 30, 40], 'max_depth': [3, 7, 9, 12]}
model = GradientBoostingRegressor(learning_rate=0.5, random_state = 77)
gridForest = GridSearchCV(estimator=model, param_grid=parameters)
gridForest.fit(X_train, y_train)
print('Best score:', round(gridForest.best_score_,3))
print('Best model:', gridForest.best_estimator_)


# Оптимальные значения параметров: n_estimators = 400, max_features = 30.

# Построим модель GradientBoostingRegressor с учетом подобранных параметров:

# In[81]:


gradient_model = GradientBoostingRegressor(learning_rate=0.5, max_features=40, n_estimators=400, random_state=77)
gradient_model.fit(X_train, y_train)


# Выводим метрики:

# In[94]:


preds = gradient_model.predict(X_test)
print('Gradient Boosting Regression Estimation:')
print('R2-score:',round(r2_score(y_test,preds),3))
print('MSE:',round(mean_squared_error(y_test,preds),3))
print('MAE:',round(mean_absolute_error(y_test,preds),3))


# Получим значимость признаков:

# In[83]:


feature_importances = gradient_model.feature_importances_
feature_importances_df = pd.DataFrame(feature_importances,
                                      columns=['Importances'],
                                      index = X.columns).sort_values('Importances', ascending=True)
feature_importances_df[-10:].plot.barh(figsize=(15,12))


# ### Вывод

# При создании модели GradientBoostingRegressor использовались следующие гиперпараметры: learning_rate=0.5, max_features=40, n_estimators=400, random_state=77. На тестовой выборке мы получили R2-score: 0.931, MSE: 2248.334, MAE: 31.384.
# 
# При анализе значимости признаков мы выяснили, что наибольшее влияние на предсказание таргета оказывают следующие признаки: atemp, workingday, hr_17, yr, hr_18.

# ### Общий вывод

# Лучше всех показала себя модель GradientBoostingRegressor с R2-score = 0.931, MSE = 2248.334, MAE = 31.384.
# 
# В моделях RandomForestRegressor и GradientBoostingRegressor были схожие самые значимые признаки (atemp, hr_17, yr, hr_18), но они имели разные веса в каждой модели.
