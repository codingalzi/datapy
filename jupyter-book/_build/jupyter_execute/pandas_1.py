#!/usr/bin/env python
# coding: utf-8

# (sec:pandas_1)=
# # 판다스 데이터프레임

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


# `Series`와 `DataFrame`을 표로 보여줄 때 사용되는 행의 수를 20으로 지정한다. 
# 기본 값은 60이다.

# In[3]:


pd.options.display.max_rows # 원래 60이 기본.


# 기본값을 20으로 변경한다.

# In[4]:


pd.set_option("display.max_rows", 20)


# **참고 자료**
# 
# 1. {ref}`sec:pandas10min_1`와 {ref}`sec:pandas10min_2`를
#     먼저 무조건 따라하며 읽을 것을 추천한다.
#     그러면 데이터프레임의 다양한 기능에 맛을 보고 여기서부터 하나씩 차분히 살펴볼 수 있다.
# 1. [판다스 공식 사용자 설명서](https://pandas.pydata.org/docs/user_guide/index.html):
#     함께 읽기를 권장한다.
#     여기서 다루는 내용과 유사한 수준으로 판다스의 보다 다양한 기능을 설명한다.

# **주요 내용**

# 판다스가 제공하는 시리즈(Series)와 데이터프레임(DataFrame)은 
# 넘파이 어레이와 유사한 기능과 함께 데이터를 조작하고 다루기 위한 다양한 기능을 추가로 제공한다.
# 판다스(pandas)는 넘파이와 함께 앞으로 가장 많이 다룰 라이브러리이며, 데이터 분석에 사용되는 많은 다른 라이브러리와 함께 자주 사용된다.
# 
# 넘파이 어레이는 수치형 데이터를 처리하는 데에 특화된 반면에 
# 판다스의 데이터프레임은 표(table) 형식의 데이터 또는 다양한 형태의 데이터를 다룰 수 있다.
# 
# 이번 장에서 소개하는 내용은 다음과 같다.
# 
# * `Series`와 `DataFrame` 객체 소개
# * `Series`와 `DataFrame`의 주요 도구: 인덱싱, 삭제, 연산, 정렬
# * 기초 통계 활용
# 
# 1편에서는 먼저 시리즈와 데이터프레임을 소개한다.

# ## 시리즈

# **시리즈**<font size='2'>Series</font>는 1차원 어레이와 동일한 구조를 갖는다. 
# 다만 인덱스<font size='2'>index</font>를 0, 1, 2 등이 아닌 임의의 값으로 지정할 수 있으며
# 항상 함께 고려해야 한다.

# ### 시리즈 생성과 인덱스

# 시리즈를 생성하기 위해 리스트, 넘파이 1차원 어레이, 사전 등을 이용할 수 있다.

# **리스트와 어레이 활용**

# 1차원 리스트 또는 어레이를 이용하여 간단하게 시리즈를 생성할 수 있다.
# 그러면 지정된 순서대로 0, 1, 2, 등의 인덱스가 자동 생성되어 함께 보여진다.
# 
# * 인덱스: 별도로 지정하지 않으면 리스트, 넘파이 어레이 등에서 사용된 인덱스가 기본으로 사용됨.
# * `dtype`: 사용된 항목의 자료형을 가리키며 모든 항목은 동일한 자료형을 가져야 함. 

# 아래 코드는 리스트를 이용하여 시리즈를 생성한다.

# In[5]:


ojb1 = pd.Series([4, 7, -5, 3])
ojb1


# 1차원 어레이도 이용할 수 있다.

# In[6]:


ojb1 = pd.Series(np.array([4, 7, -5, 3]))
ojb1


# **사전 활용**

# 사전을 이용하여 시리즈를 생성할 수 있다.
# 
# * 키 => 인덱스
# * 값 => 값

# In[7]:


sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
obj3 = pd.Series(sdata)
obj3


# 사전을 이용하더라도 인덱스를 따로 지정할 수 있다.
# 그러면 사전에 키로 사용되지 않은 인덱스는 누락되었다는 의미로 `NaN`이 표시된다.
# 또한 인덱스 리스트에 포함되지 않는 (사전의) 키는 포함되지 않는다.
# 
# * `California`: `sdata` 사전에 키로 사용되지 않았기에 `Nan`으로 지정
# * `Utah`: `states` 리스트에 포함되지 않았기에 생성된 시리즈에 사용되지 않음.

# In[8]:


states = ['California', 'Ohio', 'Oregon', 'Texas']

obj4 = pd.Series(sdata, index=states)
obj4


# 역으로 시리즈를 사전으로 변환할 수도 있다. 
# 
# * 인덱스 => 키
# * 값 => 값

# In[9]:


dict(obj4)


# :::{admonition} 사전과 시리즈 비교
# :class: info
# 
# 시리즈는 길이가 고정되었으며 항목들의 순서를 따지고 중복을 허용하는 사전으로 간주될 수 있다.
# 
# | 사전 | 시리즈 |
# | :---: | :---:  |
# | 키(key) | 인덱스 |
# | 값 | 값    |
# | 순서 없음 | 순서 중요 |
# | 중복 없음 | 중복 허용 |
# :::

# ### `index` 속성과 `index` 키워드

# 사용된 인덱스는 `index` 속성이 갖고 있다.
# 자동으로 생성된 경우 인덱스는 `range`와 유사한 `RangeIndex` 자료형이다.

# In[10]:


ojb1


# In[11]:


ojb1.index


# 기존에 사용된 인덱스를 완전히 새로운 인덱스로 대체할 수도 있다.

# In[12]:


ojb1.index = ['Bob', 'Steve', 'Jeff', 'Ryan']
ojb1


# 처음부터 인덱스를 지정하면서 시리즈를 생성할 수 있다.
# 
# * `index` 키워드 인자: 항목의 수와 동일한 길이를 갖는 리스트. 
#     리스트에 포함된 항목 순서대로 인덱스 지정.
#     
# 인덱스가 지정된 순서대로 사용됨에 주의하라.

# In[13]:


obj2 = pd.Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
obj2


# **중복 인덱스 허용**

# 인덱스를 중복해서 사용할 수도 있다.

# In[14]:


dup_labels = pd.Index(['d', 'd', 'a', 'a', 'a'])
dup_labels


# In[15]:


obj2


# In[16]:


pd.Series(obj2, index=dup_labels)


# **`Index` 객체**

# `index` 키워드로 지정된 인덱스는 `index` 속성이 가리키며 `Index` 객체로 저장된다.

# In[17]:


idx = obj2.index
idx


# 인덱스 객체는 1차원 어레이와 유사하게 동작한다.
# 예를 들어, 인덱싱과 슬라이싱은 리스트 또는 1차원 어레이의 경우와 동일하게 작동한다.

# In[18]:


idx[1]


# In[19]:


idx[1:]


# :::{admonition} 주의사항
# :class: info
# 
# `Index` 자료형은 불변<font size='2'>immutable</font> 자료형이다.
# 아래처럼 인덱싱을 이용하여 항목을 변경하려 하면 `TypeError`가 발생한다.
# 
# ```python
# idx[1] = 'd'
# ---------------------------------------------------------------------------
# TypeError                                 Traceback (most recent call last)
# Cell In [25], line 1
# ----> 1 idx[1] = 'd'
# 
# File ~\anaconda3\envs\dlp2\lib\site-packages\pandas\core\indexes\base.py:5035, in Index.__setitem__(self, key, value)
#    5033 @final
#    5034 def __setitem__(self, key, value):
# -> 5035     raise TypeError("Index does not support mutable operations")
# 
# TypeError: Index does not support mutable operations
# ```
# :::

# ### `name` 과 `values`

# **`name` 속성**

# `Series` 객체와 시리즈의 `Index` 객체 모두 `name` 속성을 이용하여
# 사용되는 값들에 대한 정보를 저장한다.
# 아래 코드 이름 두 개를 지정한다.
# 
# - 시리즈 이름은 population(인구): `name='population'`
# - 시리즈의 인덱스의 이름은 state(주 이름): `Index.name='state'`

# In[20]:


obj4.name = 'population'
obj4.index.name = 'state'
obj4


# **`values` 속성**

# `values` 속성이 시리즈의 항목으로 구성된 1차원 어레이를 가리킨다.

# In[21]:


ojb1.values


# ### 시리즈 연산

# 시리즈의 항목을 확인하는 기본 기능을 살펴 본다.

# **`in` 연산자**

# `in` 연산자는 인덱스 사용 여부를 사전 자료형의 키(key) 사용 여부와 동일한 방식으로 판단한다.

# In[22]:


'b' in obj2


# In[23]:


'e' in obj2


# **결측치 사용 여부 확인**

# `pd.isnull()` 함수는 누락된 항목은 `True`, 아니면 `False`로 지정하여 단번에 결측치가 포함되었는지 
# 여부를 확인해준다.

# In[24]:


pd.isnull(obj4)


# `pd.notnull()` 함수는 누락된 항목은 `False`, 아니면 `True`로 지정하여 단번에 결측치가 포함되었는지 
# 여부를 확인해준다.

# In[25]:


pd.notnull(obj4)


# 두 함수를 호출하면 실제로는 시리즈 객체의 메서드인 `isnull()` 또는 `notnull()`이 내부에서 호출된다.

# In[26]:


obj4.isnull()


# In[27]:


obj4.notnull()


# **`any()` 와 `all()` 메서드**

# `any()` 또는 `all()` 메서드를 활용하면 결측치 사용 여부를 단번에 알 수 있다.
# 예를 들어, `pd.isnull()` 과 `any()` 메서드의 활용 경과가 `True` 이면 결측치가 있다는 의미이다.

# In[28]:


obj4.isnull().any()


# 반면에 `pd.notnull()` 과 `all()` 메서드의 활용 경과가 `False` 이면 역시 결측치가 있다는 의미이다.

# In[29]:


obj4.notnull().all()


# 넘파이의 `np.any()`, `np.all()` 를 활용해도 동일한 결과를 얻는다.

# In[30]:


np.any(obj4.isnull())


# In[31]:


np.all(obj4.notnull())


# ## 데이터프레임

# **데이데프레임**<font size='2'>DataFrame</font>은 인덱스를 공유하는 여러 개의 시리즈를 다루는 객체다. 
# 아래 그림은 세 개의 시리즈를 하나의 데이터프레임으로 만든 결과를 보여준다.

# <img src="https://raw.githubusercontent.com/codingalzi/pydata/master/notebooks/images/series-dataframe01.png" style="width:700px;">

# 위 이미지에 있는 세 개의 시리즈는 다음과 같으며,
# `name` 속성을 이용하여 각 시리즈의 이름도 함께 지정한다.

# In[32]:


series1 = pd.Series([4, 5, 6, 3 , 1], name="Mango")
series1


# In[33]:


series2 = pd.Series([5, 4, 3, 0, 2], name="Apple")
series2


# In[34]:


series3 = pd.Series([2, 3, 5, 2, 7], name="Banana")
series3


# ### 데이터프레임 생성

# **`pd.concat()` 함수 활용**

# `pd.concat()` 함수도 여러 개의 시리즈를 묶어 하나의 데이터프레임을 생성한다.
# 단, 축을 이용하여 묶는 방식을 지정한다.
# 위 그림에서처럼 옆으로 묶으려면 열 단위로 묶는다는 의미에서 `axis=1`로 지정한다.
# 각 열의 이름은 해당 시리즈의 `name`이 가리키는 값으로 지정된다.
# 
# __참고:__ `concat`는 이어붙인다의 의미를 갖는 concatenate 영어 단어에서 유래한다.

# In[35]:


pd.concat([series1, series2, series3], axis=1)


# **2차원 넘파이 어레이 활용**

# In[36]:


data = pd.DataFrame(np.arange(16).reshape((4, 4)),
                    index=['Ohio', 'Colorado', 'Utah', 'New York'],
                    columns=['year', 'state', 'p', 'four'])
data


# **리스트 사전 활용**

# 리스트를 값으로 갖는 사전을 이용하여 데이터프레임을 생성할 수 있다.
# 
# 아래 코드에서 `data`는 `state`(주 이름), `year`(년도), `pop`(인구)을 키(key)로 사용하며,
# 해당 특성에 해당하는 데이터로 구성된 리스트를 값으로 갖는 사전 객체이다.

# In[37]:


dict2 = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada', 'NY', 'NY', 'NY'],
         'year': [2000, 2001, 2002, 2001, 2002, 2003, 2002, 2003, 2004],
         'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2, 8.3, 8.4, 8.5]}


# 위 사전 객체를 데이터프레임으로 변환하면 다음과 같다.

# In[38]:


frame2 = pd.DataFrame(dict2)
frame2


# **중첩 사전 활용**

# 데이터프레임을 생성함에 있어서의 핵심은 2차원 행렬 모양을 갖느냐이기에
# 각 열에 해당하는 값으로 리스트, 어레이, 사전, 시리즈 등이 사용될 수 있다.
# 따라서 아래 모양의 중첩 사전을 활용하여 데이터프레임을 생성할 수 있다.
# 그러면 최상위 키는 열의 이름으로, 내부에 사용된 키는 행의 인덱스로 사용된다.

# In[39]:


dict3 = {'Nevada': {2001: 2.4, 2002: 2.9},
         'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}


# 위 중첩 사전을 이용하여 데이터프레임을 생성하면 다음과 같다.
# 다만, 두 사전의 키가 다름에 주의하라. 
# 예를 들어, 2000 인덱스 행의 Nevada의 경우는 결측치로 처리된다. 

# In[40]:


frame3 = pd.DataFrame(dict3)
frame3


# 하나의 사전을 시리즈로 본다면 여러 개의 사전으로 이루어진 사전을 이용하여 
# 생성된 데이터프레임은 여러 개의 시리즈를 이어붙여서 생성한 데이터프레임으로 간주할 수 있다.
# 실제로 아래 두 개의 시리즈를 합친 결과와 동일함을 확인할 수 있다.

# In[41]:


nevada = pd.Series({2001: 2.4, 2002: 2.9}, name="Nevada")
nevada


# In[42]:


ohio = pd.Series({2000: 1.5, 2001: 1.7, 2002: 3.6}, name="Ohio")
ohio


# 이제 두 시리즈를 합치면 위 결과와 동일하다. 
# 단, 행 인덱스가 정렬되서 보이는 점만 조금 다르다.
# 행과 열을 기준으로 정렬하는 작업은 나중에 설명한다.

# In[43]:


pd.concat([nevada, ohio], axis=1)


# ### `name` 과 `values`

# **`name` 속성**

# 시리즈의 경우와 동일한 방식으로 행과 열의 이름을 지정할 수 있다.

# In[44]:


frame3.index.name = 'year'      # 행 이름 지정
frame3.columns.name = 'state'   # 열 이름 지정
frame3


# **`values` 속성**

# 항목들로 이루어진 2차원 어레이는 `values` 속성이 가리킨다.

# In[45]:


frame3.values


# In[46]:


frame2.values


# ### `columns` 와 `index`

# **`columns` 속성**

# `columns` 속성을 이용하여 열의 순서를 지정할 수 있다.

# In[47]:


pd.DataFrame(dict2, columns=['year', 'state', 'pop'])


# 새로운 열을 추가할 수도 있다.
# 이름만 지정할 경우 항목은 모두 `NaN`으로 처리된다.

# In[48]:


frame2 = pd.DataFrame(dict2, columns=['year', 'state', 'pop', 'debt'])
frame2


# 또는 사전에 항목을 추가하듯이 진행할 수도 있다.
# 
# __주의사항:__ `frame2`와의 충돌을 피하기 위해 복사해서 사용한다.

# In[49]:


frame2_ = frame2.copy()
frame2_["debt2"] = np.linspace(0, 1, 9)   # 구간 [0, 1]을 8개의 구간으로 쪼개기

frame2_


# `columns` 속성을 확인하면 다음과 같다.

# In[50]:


frame2.columns


# **`index` 속성**

# 인덱스를 지정하려면 `index` 속성을 이용한다.

# In[51]:


frame2 = pd.DataFrame(dict2, index=['one', 'two', 'three', 'four',
                             'five', 'six', 'seven', 'eight', 'nine'])
frame2


# 물론 `columns`, `index` 등 여러 속성을 동시에 지정할 수도 있다.

# In[52]:


frame2 = pd.DataFrame(dict2, columns=['year', 'state', 'pop', 'debt'],
                      index=['one', 'two', 'three', 'four',
                             'five', 'six', 'seven', 'eight', 'nine'])
frame2


# **중복 인덱스 허용**

# 인덱스를 중복해서 사용할 수도 있다.

# In[53]:


dup_labels = pd.Index(['one', 'two', 'two', 'three', 'three', 'three'])
dup_labels


# In[54]:


frame2


# In[55]:


pd.DataFrame(frame2, index=dup_labels)


# **`Index` 객체**

# 시리즈와 데이터프레임의 `index` 와 `columns` 속성에
# 저장된 값은 `Index` 객체다.

# In[56]:


obj = pd.Series(range(3), index=['a', 'b', 'c'])
index = obj.index
index


# In[57]:


frame3.columns


# ### 데이터프레임 연산

# 데이터프레임의 항목을 확인하는 기본 기능을 살펴 본다.

# **`in` 연산자**

# 인덱스와 열에 대한 특정 이름의 사용 여부는 `in` 연산자를 이용하여 확인한다.

# In[58]:


frame2


# In[59]:


'year' in frame2.columns


# In[60]:


'ten' in frame2.index


# **`head()` 메서드**

# `head()` 메서드는 지정된 크기만큼의 행을 보여준다. 
# 인자를 지정하지 않으면 처음 5개의 행을 보여준다.

# In[61]:


frame2.head(3)


# In[62]:


frame2.head()


# **`tail()` 메서드**

# `tail()` 메서드는 지정된 크기만큼의 행을 뒤에서부터 보여준다. 
# 인자를 지정하지 않으면 뒤에서부터 5개의 행을 보여준다.

# In[63]:


frame2.tail(3)


# In[64]:


frame2.tail()


# **전치 데이터프레임**

# 2차원 행렬의 전치 행렬처럼 전치 데이터프레임은 행과 열의 위치를 바꾼 결과이다.
# 당연히 행과 열에 사용된 이름이 적절하게 전치된다.

# In[65]:


frame3


# In[66]:


frame3.T


# **결측치 사용 여부 확인**

# In[67]:


frame2


# `isnull()` 메서드는 누락된 항목은 `True`, 아니면 `False`로 지정하여 단번에 결측치가 포함되었는지 
# 여부를 확인해준다.

# In[68]:


frame2.isnull()


# `notnull()` 메서드는 누락된 항목은 `False`, 아니면 `True`로 지정하여 단번에 결측치가 포함되었는지 
# 여부를 확인해준다.

# In[69]:


frame2.notnull()


# **`any()` 와 `all()` 메서드**

# `any()` 또는 `all()` 메서드를 활용하면 결측치 사용 여부를 단번에 알 수 있다.
# 두 메서드는 기본적으로 열별로 적어도 하나가 또는 모두 참인지 여부를 확인한다.
# 결과는 시리즈다.

# In[70]:


frame2.isnull().any()


# 축 키워드 인자를 `axis=0`로 지정한 것과 동일하다.

# In[71]:


frame2.isnull().any(axis=0)


# 행 별로 결측치 존재 여부를 확인하려면 축 키워드 인자를 `axis=1`로 지정한다.
# 결과는 시리즈다.

# In[72]:


frame2.isnull().any(axis=1)


# 반면에 `all()` 메서드는 열별 또는 행별로 모두 참인지 확인한다.

# In[73]:


frame2.isnull().all()


# 축 키워드 인자를 `axis=0`로 지정한 것과 동일하다.

# In[74]:


frame2.isnull().all(axis=0)


# 행 별로 결측치 존재 여부를 확인하려면 축 키워드 인자를 `axis=1`로 지정한다.
# 결과는 시리즈다.

# In[75]:


frame2.isnull().all(axis=1)


# 넘파이의 `np.any()`, `np.all()`는 전체 항목을 대상으로만 작동한다.

# In[76]:


np.any(frame2.isnull())


# In[77]:


np.all(frame2.notnull())


# 데이터프레임의 `any()` 메서드와 `all()` 메서드를 이용하여 전체 항목을 확인하려면 해당 메서드를 두 번 적용해야 한다.

# In[78]:


frame2.isnull().any().any()


# In[79]:


frame2.isnull().any(axis=1).any()


# In[80]:


frame2.isnull().all().all()


# In[81]:


frame2.isnull().all(axis=1).all()


# ## 참고 자료

# 1. [Pandas Tutor: Using Pyodide to Teach Data Science at Scale](https://blog.pyodide.org/posts/pandastutor/): 데이터프레임 기능 시각화 도구
