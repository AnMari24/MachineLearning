#!/usr/bin/env python
# coding: utf-8

# ## Лабораторная работа № 4
# ### Андрюшина Мария, 932001
# ### Аренда велосипедов

# In[103]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN


# In[3]:


import warnings
warnings.filterwarnings('ignore')


# ### Работа с данными

# In[34]:


df = pd.read_csv('hour.csv', index_col = 0)
df.head()


# In[35]:


df.describe()


# In[36]:


df.info()


# In[37]:


df.isnull().sum()/df.shape[0]


# Пропущенных значений нет

# In[40]:


df['dteday'] = pd.to_datetime(df['dteday'])
df['day'] = df['dteday'].dt.day
df = df.drop(['dteday'], axis=1)


# Построим тепловую карту для всех признаков:

# In[41]:


plt.figure(figsize=(13, 12))
sns.heatmap(df.corr(), annot = True, square=True)


# Возьмём два признака: hum и atemp

# In[45]:


features = ['hum', 'atemp']


# In[46]:


X = df.copy()
X = X[features]
X


# In[47]:


min_max_scaler = MinMaxScaler()
X = min_max_scaler.fit_transform(X)


# In[49]:


X = pd.DataFrame(X, columns=[features])
X.head()


# ### K-means

# Подберём оптимальное количество кластеров:

# In[63]:


df_n_kmeans = pd.DataFrame(columns=['n', 'score'])
for i in range(2, 9):
    kmeans = KMeans(n_clusters=i, random_state=77)
    preds = kmeans.fit_predict(X)
    score = round(silhouette_score(X, preds), 3)
    print('Количество кластеров:', i)
    print('Коэффициент силуэта:', score)
    df_n_kmeans.loc[len(df_n_kmeans.index)] = [i, score] 
df_n_kmeans


# In[64]:


plt.figure(figsize=(12,8))
plt.plot(df_n_kmeans['n'].values,df_n_kmeans['score'].values, color='orange')
plt.legend()
plt.show()


# Оптимальное количество кластеров равно 3.

# Подберем параметр n_init:

# In[143]:


df_init_kmeans = pd.DataFrame(columns=['n', 'score'])
for i in range (6, 15, 2):
    kmeans = KMeans(n_clusters=3, n_init = i, random_state=77)
    preds = kmeans.fit_predict(X)
    score = round(silhouette_score(X, preds), 5)
    print(i)
    print('Коэффициент силуэта:', score)
    df_init_kmeans.loc[len(df_init_kmeans.index)] = [i, score] 
df_init_kmeans


# Оптимальное значение для параметра n_init равно 10.

# Построим модель KMeans с учетом подобранных параметров:

# In[88]:


kmeans = KMeans(n_clusters = 3, n_init = 10, random_state = 77)


# In[89]:


preds = kmeans.fit_predict(X)


# In[90]:


score = silhouette_score(X, preds)
print('Коэффициент силуэта: ', score)


# In[91]:


kmeans.cluster_centers_


# In[92]:


kmeans.fit(X)


# Построим центроиды кластеров на графике:

# In[108]:


plt.figure(figsize=(10,6))
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker = 'X', c = 'red', s = 100, label = 'Centroids')
plt.title('K-Means Clustering')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend()
plt.show()


# Построим на графике кластеры:

# In[102]:


plt.figure(figsize = (10,6))
plt.scatter(X.iloc[:,0], X.iloc[:,1], c = kmeans.labels_, cmap = 'spring')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 150, c = 'green', marker = 'X', label = 'Centroids')
plt.legend(loc = 'upper right')


# ### Вывод

# При создании модели KMeans использовались следующие гиперпараметры: n_clusters = 3, n_init = 10, random_state = 77. Коэффициент силуэта равен 0.415. На графике выделены три кластера примерно одинакового размера. Расстояние между центроидами кластеров  примерно одинаковое. 

# ### Аггломеративная кластеризация

# Подберём оптимальное количество кластеров:

# In[106]:


df_n_agglom_clust = pd.DataFrame(columns=['n', 'score'])
for i in range(2, 9):
    agglom_clust = AgglomerativeClustering(n_clusters=i)
    preds = agglom_clust.fit_predict(X)
    score = round(silhouette_score(X, preds), 3)
    print('Количество кластеров:', i)
    print('Коэффициент силуэта:', score)
    df_n_agglom_clust.loc[len(df_n_agglom_clust.index)] = [i, score] 
df_n_agglom_clust


# In[109]:


plt.figure(figsize=(12,8))
plt.plot(df_n_agglom_clust['n'].values,df_n_agglom_clust['score'].values, color='orange')
plt.legend()
plt.show()


# Оптимальное количество кластеров равно 2.

# Построим модель AgglomerativeClustering с учетом подобранных параметров:

# In[148]:


agglom_clust = AgglomerativeClustering(n_clusters=2)
agglom_clust.fit(X)


# In[149]:


agglom_clust_labels = agglom_clust.labels_


# In[150]:


score = silhouette_score(X, agglom_clust.fit_predict(X))
print('Коэффициент силуэта: ', score)


# Построим на графике кластеры:

# In[151]:


plt.figure(figsize=(12,8))
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=ward_labels, cmap='winter')
plt.title("Agglomerative Clustering")
plt.show()


# ### Вывод

# При создании модели AgglomerativeClustering использовались следующие гиперпараметры: n_clusters=2. Коэффициент силуэта равен 0.365. На графике выделены два кластера примерно одинакового размера.

# ### DBScan

# Подберём оптимальные параметры модели:

# In[137]:


score1 = 0
for eps in np.arange(0.1, 0.5, 0.1):
    for min_samples in range(1, 5):
        dbscan = DBSCAN(eps = eps, min_samples = min_samples, n_jobs = -1)
        preds = dbscan.fit_predict(X)
        n_clusters = len(set(preds)) - (1 if -1 in preds else 0)
        if n_clusters < 2:
            continue
        score = silhouette_score(X, preds)
        if score > score1:
            score1 = score
            eps1 = eps
            min_samples1 = min_samples
print('eps: ', eps1)
print('min_samples: ', min_samples1)
print('Коэффициент силуэта: ', score1)


# Оптимальные значения параметров: eps = 0.1, min_samples = 1.

# Построим модель DBSCAN с учетом подобранных параметров:

# In[139]:


dbscan = DBSCAN(eps = 0.1, min_samples = 1)
preds = dbscan.fit_predict(X)


# In[140]:


score = silhouette_score(X, preds)
print('Коэффициент силуэта: ', score)


# Построим на графике кластеры:

# In[158]:


plt.figure(figsize=(12,8))
plt.scatter(X.iloc[:,0], X.iloc[:,1], c=clusters, cmap='Dark2')
plt.show()


# ### Вывод

# При создании модели DBSCAN использовались следующие гиперпараметры: eps = 0.1, min_samples = 1. Коэффициент силуэта равен 0.457. На графике выделены два кластера. Первый имеет меньший размер, но при этом существенно отдалён от второго.

# ### Общий вывод

# Лучше всех показала себя модель DBSCAN с коэффициентом силуэта, равным 0.457. При этом число элементов в первом кластере сильно отличается от числа элементов во втором.
# 
# Модели K-means и AgglomerativeClustering выделяли кластеры более равномерно. В каждом кластере получилось примерно одинаковое количество элементов.
