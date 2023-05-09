#!/usr/bin/env python
# coding: utf-8

# (sec:titanic)=
# # 판다스 실전 활용: 타이타닉

# **주요 내용**

# 타이타닉<font size='2'>Titanic</font> 데이터셋을 데이터프레임으로 불러와서 전처리 하는 과정을 살펴 본다.

# **기본 설정**

# `pandas` 라이브러리는 보통 `pd` 라는 별칭으로 사용된다.

# In[1]:


import pandas as pd
import numpy as np


# 랜덤 시드, 어레이 내부에 사용되는 부동소수점 정확도, 도표 크기 지정 옵션 등은 이전과 동일하다.

# In[2]:


np.random.seed(12345)
np.set_printoptions(precision=4, suppress=True)

import matplotlib.pyplot as plt
plt.rc('figure', figsize=(10, 6))


# 사이킷런<font size='2'>scikit-learn</font> 라이브러리를 일부 이용한다.

# In[3]:


from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


# **참고**: 
# 
# - https://jaketae.github.io/study/sklearn-pipeline/

# In[4]:


X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)
X.drop(['boat', 'body', 'home.dest'], axis=1, inplace=True)


# In[5]:


X.head()


# In[6]:


X.isnull().any()


# In[7]:


X.isnull().sum()


# In[8]:


X.isnull().sum()/len(X) * 100


# In[9]:


X.drop(['cabin'], axis=1, inplace=True)


# In[10]:


X.isnull().sum()


# In[11]:


import pandas as pd
import seaborn as sns

X_comb = pd.concat([X, y.astype(float)], axis=1)
g = sns.heatmap(X_comb[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'survived']].corr(),
                annot=True, 
                cmap = "coolwarm")


# In[12]:


X['family_size'] = X['parch'] + X['sibsp']
X.drop(['parch', 'sibsp'], axis=1, inplace=True)
X['is_alone'] = 1
X['is_alone'].loc[X['family_size'] > 1] = 0

X.head()


# In[13]:


X['title'] =  X['name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
X.drop(["name"], axis=1, inplace=True)

X.head()


# In[14]:


pd.crosstab(X['title'], X['sex'])


# In[15]:


print(f"Miss: {np.sum(y.astype(int)[X.title == 'Miss']) / len(X.title == 'Miss')}")
print(f"Mrs: {np.sum(y.astype(int)[X.title == 'Mrs']) / len(X.title == 'Mrs')}")


# In[16]:


rare_titles = (X['title'].value_counts() < 10)
rare_titles


# In[17]:


X.title.loc[X.title == 'Miss'] = 'Mrs'
X['title'] = X.title.apply(lambda x: 'rare' if rare_titles[x] else x)


# In[18]:


X.drop('ticket', axis=1, inplace=True)

X.head()


# In[19]:


X.dtypes


# **참고**: 
# 
# - https://www.jcchouinard.com/classification-machine-learning-project-in-scikit-learn/

# In[20]:


X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)
X.drop(['boat', 'body', 'home.dest'], axis=1, inplace=True)


# In[21]:


from sklearn.datasets import fetch_openml
 
titanic = fetch_openml('titanic', version=1, as_frame=True)


# In[22]:


type(titanic)


# In[23]:


df = titanic['data']
df.head()


# In[24]:


df.columns


# In[25]:


df['survived'] = titanic['target']


# In[26]:


df.head()


# In[27]:


df.describe()


# **결측치 확인과 시각화**

# In[28]:


df.info()


# In[29]:


df.isnull().sum()


# In[30]:


miss_vals = pd.DataFrame(df.isnull().sum() / len(df) * 100)
miss_vals


# [`seaborn.set_theme()` 함수](https://seaborn.pydata.org/generated/seaborn.set_theme.html)를 이용하면 보다 세련된 그래프를 그린다.

# In[31]:


sns.set_theme()


# In[32]:


miss_vals.plot(kind='bar',
               title='Missing values in percentage',
               ylabel='percentage'
              )
 
plt.show()


# **타깃 시각화**

# In[33]:


df.survived.value_counts().plot(kind='bar')
 
plt.xlabel('Survival')
plt.ylabel('# of passengers')
plt.title('Number of passengers based on their survival')
plt.show()


# **연령별 생존자**

# In[34]:


df.age.dropna()


# In[35]:


df['age'][df.survived == '1'].dropna()


# In[36]:


fig, ax = plt.subplots()
 
ax.hist(df.age.dropna(), label='Not survived')
ax.hist(df['age'][df.survived == '1'].dropna(), label='Survived')
 
plt.ylabel('Survivors')
plt.xlabel('Age')
plt.title('Survival by age')
plt.legend()
plt.show()


# **성별 생존률**

# In[37]:


((df['survived'][df.sex == 'male']) == 1).sum()


# In[38]:


161/843


# In[39]:


df['survived'] = df.survived.astype('int')
 
sns.barplot(data=df, 
            x='sex',
            y='survived'
           )
 
plt.title('Survival by gender')
plt.show()


# **참고: `sns.barplot()` 함수**

# In[40]:


df_1 = sns.load_dataset("penguins")
sns.barplot(data=df_1, x="island", y="body_mass_g")


# In[41]:


df_1


# In[42]:


sns.barplot(data=df_1, x="island", y="body_mass_g", hue="sex")


# - `errorbar` 옵션 인자
#     - `ci`: confidence interval
#     - `pi`: percentile interval
#     - `se`: standard error
#     - `sd`: standard deviation
#     
# - 참고: [Statistical estimation and error bars](https://seaborn.pydata.org/tutorial/error_bars.html)

# In[43]:


sns.barplot(data=df_1, x="island", y="body_mass_g", errorbar="sd") # 표준편차


# **신분별 생존자**

# In[44]:


sns.countplot(x='pclass', data=df)
plt.title('Unique survivors by class')
plt.show()


# In[45]:


sns.barplot(x='pclass', y='survived', data=df)
plt.title('Percent survivers by class')
plt.show()


# **출발 항구별 생존률**

# In[46]:


sns.barplot(x='embarked', y='survived', data=df)
plt.title('Percent survivers by port of embarkation')
plt.show()


# **생존자 특성 분리**

# In[47]:


X = df.drop('survived', axis=1)
y = df['survived']


# **데이터 전처리**

# In[48]:


X['family'] = X['sibsp'] + X['parch']
X.loc[X['family'] > 0, 'travelled_alone'] = 0
X.loc[X['family'] == 0, 'travelled_alone'] = 1
X.drop(['family', 'sibsp', 'parch'], axis=1, inplace=True)
sns.countplot(x='travelled_alone', data=X)
plt.title('Number of passengers travelling alone')
plt.show()


# ### Preprocess Data with Scikit-learn

# #### 결측치 처리

# In[49]:


from sklearn.impute import SimpleImputer
 
def get_parameters(df):
    parameters = {}
    for col in df.columns[df.isnull().any()]:
        if df[col].dtype == 'float64' or df[col].dtype == 'int64' or df[col].dtype =='int32':
            strategy = 'mean'
        else:
            strategy = 'most_frequent'
        missing_values = df[col][df[col].isnull()].values[0]
        parameters[col] = {'missing_values':missing_values, 'strategy':strategy}
    return parameters
 
parameters = get_parameters(X)
 
for col, param in parameters.items():
    missing_values = param['missing_values']
    strategy = param['strategy']
    imp = SimpleImputer(missing_values=missing_values, strategy=strategy)
    X[col] = imp.fit_transform(X[[col]])
 
X.isnull().sum()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# **참고**: 
# 
# - https://medium.datadriveninvestor.com/implementation-of-data-preprocessing-on-titanic-dataset-6c553bef0bc6

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




