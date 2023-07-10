#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# In[2]:


boston_url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ST0151EN-SkillsNetwork/labs/boston_housing.csv'
boston_df=pd.read_csv(boston_url)


# In[3]:


boston_df.describe()


# In[4]:


boston_df.head(15)


# In[5]:


# Diagrama de caja para el valor medio de las viviendas ocupadas por el propietario (MEDV)
plt.figure(figsize=(8, 6))
sns.boxplot(data=boston_df, y='MEDV')
plt.title('Valor medio de las viviendas ocupadas por el propietario')
plt.show()


# In[6]:


# Gráfico de barras para la variable del río Charles (CHAS)
plt.figure(figsize=(8, 6))
boston_df['CHAS'].value_counts().plot(kind='bar')
plt.title('Variable del Rió Charles (Número de casas cercas / lejos de)')
plt.xlabel('CHAS')
plt.ylabel('Count')
plt.show()


# In[7]:


# Discretizar la variable de edad (AGE) en tres grupos
bins = [0, 35, 70, float('inf')]
labels = ['<=35', '35-70', '>70']
boston_df['AGE_Group'] = pd.cut(boston_df['AGE'], bins=bins, labels=labels)


# In[8]:


# Diagrama de caja para MEDV frente a AGE_Group
plt.figure(figsize=(8, 6))
sns.boxplot(data=boston_df, x='AGE_Group', y='MEDV')
plt.title('Valor medio de las viviendas ocupadas por sus propietarios según la edad')
plt.xlabel('AGE Group')
plt.ylabel('MEDV')
plt.show()


# In[9]:


# Diagrama de dispersión para NOX y proporción de acres comerciales no minoristas (INDUS)
plt.figure(figsize=(8, 6))
plt.scatter(boston_df['NOX'], boston_df['INDUS'])
plt.title('Concentración de óxidos nítricos vs Proporción de acres comerciales no minoristas')
plt.xlabel('NOX')
plt.ylabel('INDUS')
plt.show()


# In[10]:


# Histograma para la variable de proporción de alumnos por profesor (PTRATIO)
plt.figure(figsize=(8, 6))
plt.hist(boston_df['PTRATIO'], bins=10)
plt.title('Proporción de alumnos por profesor')
plt.xlabel('PTRATIO')
plt.ylabel('Count')
plt.show()


# In[11]:


# Prueba T para muestras independientes para comparar el valor medio de las casas delimitadas por el río Charles o no
charles_houses = boston_df[boston_df['CHAS'] == 1]['MEDV']
non_charles_houses = boston_df[boston_df['CHAS'] == 0]['MEDV']
t_statistic, p_value = stats.ttest_ind(charles_houses, non_charles_houses)
alpha = 0.05


# In[12]:


print("Prueba T para muestras independientes:")
print(f"T-statistic: {t_statistic}")
print(f"P-value: {p_value}")


# In[13]:


if p_value < alpha:
    print("Conclusión: Hay diferencia significativa en el valor medio de las casas delimitadas por el río Charles.")
else:
    print("Conclusión: No hay diferencia significativa en el valor medio de las casas delimitadas por el río Charles.")


# In[14]:


# ANOVA para comparar los valores medios de las casas (MEDV) para cada proporción de unidades ocupadas por el propietario construidas antes de 1940 (AGE)
groups = [group['MEDV'] for name, group in boston_df.groupby('AGE_Group')]
f_statistic, p_value = stats.f_oneway(*groups)


# In[15]:


print("\nANOVA:")
print(f"F-statistic: {f_statistic}")
print(f"P-value: {p_value}")


# In[16]:


if p_value < alpha:
    print("Conclusión: Existe diferencia significativa en los valores medios de las casas (MEDV) para cada proporción de unidades ocupadas por el propietario construidas antes de 1940.")
else:
    print("Conclusión: No existe diferencia significativa en los valores medios de las casas (MEDV) para cada proporción de unidades ocupadas por el propietario construidas antes de 1940.")


# In[17]:


# Correlación de Pearson entre las concentraciones de óxido nítrico y la proporción de acres comerciales no minoristas
pearson_corr, p_value = stats.pearsonr(boston_df['NOX'], boston_df['INDUS'])


# In[18]:


print("\nCorrelación de Pearson:")
print(f"Pearson correlation coefficient: {pearson_corr}")
print(f"P-value: {p_value}")


# In[19]:


if p_value < alpha:
    print("Conclusión: Existe una correlación significativa entre las concentraciones de óxido nítrico y la proporción de acres comerciales no minoristas.")
else:
    print("Conclusión: No existe una correlación significativa entre las concentraciones de óxido nítrico y la proporción de acres comerciales no minoristas.")


# In[20]:


# Análisis de regresión para la distancia ponderada adicional a los cinco centros de empleo de Boston y el valor medio de las viviendas ocupadas por sus propietarios
slope, intercept, r_value, p_value, std_err = stats.linregress(boston_df['DIS'], boston_df['MEDV'])


# In[21]:


print("\nAnálisis de regresión:")
print(f"Slope: {slope}")
print(f"Intercept: {intercept}")
print(f"R-value: {r_value}")
print(f"P-value: {p_value}")
print(f"Standard Error: {std_err}")


# In[ ]:




