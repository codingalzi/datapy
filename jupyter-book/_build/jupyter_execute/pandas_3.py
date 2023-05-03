#!/usr/bin/env python
# coding: utf-8

# (sec:pandas_3)=
# # 데이터프레임 핵심 2부

# **주요 내용**

# `Series`와 `DataFrame` 객체를 다루는 다양한 도구를 살펴본다.
# 
# * 연산
# * 정렬
# * `drop()` 메서드: 행/열 삭제
# * `fillna()` 메서드: 결측치 처리
# * `reset_index()` 메서드: 인덱스 초기화

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


# ## 연산

# ### 산술 연산

# 시리즈/데이터프레임의 사칙 연산은 아래 원칙을 따르면서 항목별로 이뤄진다.
# 
# - 연산에 사용된 모든 행과 열의 라벨 모두 포함
# - 공통으로 사용되는 행과 열의 라벨에 대해서만 연산 적용. 그렇지 않으면 `NaN`으로 처리.
# 
# 브로드캐스팅은 넘파이 어레이 연산처럼 필요한 경우 자동 적용된다.

# **시리즈 산술 연산**

# In[5]:


s1 = pd.Series([7.3, -2.5, 3.4, 1.5], index=['a', 'c', 'd', 'e'])
s1


# In[6]:


s1*2


# In[7]:


s2 = pd.Series([-2.1, 3.6, -1.5, 4, 3.1],
               index=['a', 'c', 'e', 'f', 'g'])
s2


# In[8]:


s1 + s2


# **데이터프레임 산술 연산**

# In[9]:


df1 = pd.DataFrame(np.arange(9.).reshape((3, 3)), 
                   columns=list('bcd'),
                   index=['Ohio', 'Texas', 'Colorado'])
df1


# In[10]:


df1 - 3


# In[11]:


df2 = pd.DataFrame(np.arange(12.).reshape((4, 3)), columns=list('bde'),
                   index=['Utah', 'Ohio', 'Texas', 'Oregon'])
df2


# In[12]:


df1 + df2


# ### 연산과 결측치

# 공통 인덱스가 아니거나 결측치가 이미 존재하는 경우 기본적으로 결측치로 처리된다.
# 하지만 `fill_value` 키워드 인자를 이용하여 지정된 값으로 처리하게 만들 수도 있다.
# 다만, 연산 기호 대신에 해당 연산의 메서드를 활용해야 한다.

# In[13]:


df1.add(df2, fill_value=0)


# 기본적으로 사용되는 산술 연산 기호에 해당하는 메서드가 존재한다.
# 아래는 가장 많이 사용되는 연산 메서드들이다.
# 
# | 메서드 | 설명 |
# | :--- | :--- |
# | `add()` | 덧셈(`+`) 계산 메서드 | 
# | `sub()` | 뺄셈(`-`) 계산 메서드 | 
# | `mul()` | 곱셈(`*`) 계산 메서드 | 
# | `div()` | 나눗셈(`/`) 계산 메서드 | 
# | `floordiv()` | 몫 (`//`) 계산 메서드 | 
# | `pow()` | 거듭제곱(`**`) 메서드 | 

# ### 브로드캐스팅

# 넘파이에서 2차원 어레이와 1차원 어레이 사이에
# 브로드캐스팅이 가능한 경우,
# 즉, 차원을 맞출 수 있는 경우에 연산이 가능했다.

# In[14]:


arr = np.arange(12.).reshape((3, 4))
arr


# In[15]:


arr[0]


# In[16]:


arr - arr[0]


# 브로드캐스팅이 불가능하면 오류가 발생한다.

# In[17]:


arr[:,1]


# In[18]:


try:
    arr + arr[:, 1]
except:
    print("브로드캐스팅 불가능!")


# 물론 아래와 같이 브로드캐스팅이 가능하도록 모양을 변환한 다음엔 연산이 가능하다.

# In[19]:


arr_1 = arr[:,1][:, np.newaxis]
arr_1


# In[20]:


arr + arr_1


# 데이터프레임과 시리즈 사이의 연산도 동일하게 작동한다.
# 다만, 행 또는 열에 대한 연산 여부를 확실하게 구분해주어야 한다.

# In[21]:


frame = pd.DataFrame(np.arange(12.).reshape((4, 3)),
                     columns=list('bde'),
                     index=['Utah', 'Ohio', 'Texas', 'Oregon'])
frame


# In[22]:


series = frame.iloc[0]
series


# 브로드캐스팅은 기본족으로 행 단위로 이루어진다.
# 따라서 아래처럼 데이터프레임과 시리즈의 연산을 그냥 적용할 수 있다.

# In[23]:


frame - series


# 공통 인덱스가 존재하면 두 인자 모두에 대해 브로드캐스팅이 적용된다.

# In[24]:


series2 = pd.Series(range(3), index=['b', 'e', 'f'])
series2


# In[25]:


frame + series2


# 열 단위로 데이터프레임과 시리즈를 더하려면 
# 해당 연산 메서드를 `axis=0` 키워드 인자와 함께 적용해야 한다.

# In[26]:


series3 = frame['d']
series3


# In[27]:


frame.sub(series3, axis=0)


# `axis='index'`를 사용해도 된다.

# In[28]:


frame.sub(series3, axis='index')


# ### 유니버설 함수

# 유니버설 함수는 넘파이의 경우와 동일하게 작동한다.

# In[29]:


frame = pd.DataFrame(np.random.randn(4, 3), columns=list('bde'),
                     index=['Utah', 'Ohio', 'Texas', 'Oregon'])
frame


# 넘파이의 `abs()` 함수를 적용하면 항목별로 이루어진다.

# In[30]:


np.abs(frame)


# 시리즈에 대해서도 동일하다.

# In[31]:


np.abs(frame['b'])


# **`map()`과 `applymap()` 메서드**

# 유니버설 함수가 아닌 함수를 시리즈의 항목별로 적용하려면
# `map()` 메서드를 이용한다.
# 
# 예를 들어 아래 람다(lambda) 함수는 부동소수점을 소수점 이하 셋째 자리에서 반올림한 값만 보여주도록 
# 한다.

# In[32]:


format = lambda x: '%.2f' % x


# 시리즈에 적용해보자.

# In[33]:


frame['e'].map(format)


# 유니버설 함수가 아닌 함수를 데이터프레임의 항목별로 적용하려면
# `applymap()` 메서드를 이용한다.

# In[34]:


frame.applymap(format)


# **`apply()` 메서드**

# 행 또는 열 단위로 함수를 적용하려면 `apply()` 메서드를 활용한다.
# 기본은 열 단위로 함수가 적용되며 반환값이 스칼라 값이면 시리즈가 반환된다.
# 
# 예를 들어 아래 함수는 최댓값과 최소값의 차이를 반환한다.

# In[35]:


f1 = lambda x: x.max() - x.min()


# 데이터프레임에 적용하면 열 별로 최댓값과 최솟값의 차이를 계산하여 시리즈로 반환한다.

# In[36]:


frame.apply(f1)


# 행 별로 함수를 적용하려면 `axis=1` 또는 `axis='columls'`를 지정해야 한다.

# In[37]:


frame.apply(f1, axis='columns')


# In[38]:


frame.apply(f1, axis=1)


# 함수의 반환값이 시리즈이면 `apply()` 메서드는 데이터프레임을 반환된다.
# 예를 들어 아래 함수는 최솟값과 최댓값을 갖는 시리즈를 반환한다.

# In[39]:


def f2(x):
    return pd.Series([x.min(), x.max()], index=['min', 'max'])


# `apply()` 메서드와 함께 호출하면 열 별로 최댓값과 최솟값을 계산하여 데이터프레임으로 반환한다.

# In[40]:


frame.apply(f2)


# __참고:__ 시리즈 객체 또한 `apply()` 메서드를 갖는다. 하지만 
# 기본적으로 `map()` 메서드처럼 작동한다. 
# `map()` 메서드보다 좀 더 다야한 기능을 갖지만 여기서는 다루지 않는다.

# ## 정렬

# 행과 열의 인덱스 또는 항목을 대상으로 정렬할 수 있다.

# **`sort_index()` 메서드**

# 시리즈의 경우 인덱스를 기준으로 정렬한다. 

# In[41]:


obj = pd.Series(range(4), index=['d', 'a', 'b', 'c'])
obj


# In[42]:


obj.sort_index()


# 내림차순으로 정렬하려면 `ascending=False` 키워드 인자를 함께 사용한다.

# In[43]:


obj.sort_index(ascending=False)


# 데이터프레임의 경우 행 또는 열의 인덱스를 기준으로 정렬한다. 

# In[44]:


frame = pd.DataFrame(np.arange(8).reshape((2, 4)),
                     index=['three', 'one'],
                     columns=['d', 'a', 'b', 'c'])
frame


# 기본은 행의 인데스를 기준으로 정렬한다.

# In[45]:


frame.sort_index()


# 열의 인덱스를 기준으로 정렬하려면 `axis=1` 또는 `axis='columns'` 키워드 인자를 사용한다.

# In[46]:


frame.sort_index(axis=1)


# In[47]:


frame.sort_index(axis='columns')


# 내림차순으로 정렬하려면 `ascending=False` 키워드 인자를 함께 사용한다.

# In[48]:


frame.sort_index(axis=1, ascending=False)


# **`sort_values()` 메서드**

# 지정된 열 또는 행에 속한 값들을 기준으로 정렬할 때 사용한다.

# In[49]:


obj = pd.Series([4, 7, -3, 2])
obj


# In[50]:


obj.sort_values()


# 결측치는 맨 나중에 위치시킨다.

# In[51]:


obj = pd.Series([4, np.nan, 7, np.nan, -3, 2])
obj


# In[52]:


obj.sort_values()


# 데이터프레임의 경우 `by` 키워드 인자를 이용하여 열의 라벨을 지정해야 한다.

# In[53]:


frame = pd.DataFrame({'b': [4, 7, -3, 2], 'a': [0, 1, 0, 1]})
frame


# 예를 들어 `b` 열의 값을 기준으로 정렬한다.
# 물론 동일한 행의 값은 함께 움직인다.

# In[54]:


frame.sort_values(by='b')


# 여러 열의 값을 기준으로 정렬하려면 라벨의 리스트를 입력한다.
# 그러면 리스트 항목 순서대로 기준이 정해진다.
# 
# 예를 들어 아래 코드는 먼저 `a` 열의 항목들을 순서대로 정렬한 다음에
# 동등한 값의 경우에는 `b` 열의 항목들 순서대로 정렬한다.

# In[55]:


frame.sort_values(by=['a', 'b'])


# __참고:__ `axis=1`을 이용하여 특정 행의 값을 기준으로 정렬할 수도 있다.
# 하지만 데이터프레임은 서로 다른 특성을 열 단위로 담는 목적으로
# 사용되기에 이런 정렬은 사용할 이유가 별로 없다.

# In[56]:


frame.sort_values(by=2, axis=1, ascending=False)


# ## 행/열 삭제: `drop()` 메서드

# 특정 행 또는 열의 인덱스를 제외한 나머지로 이루어진 시리즈/데이터프레임을 생성할 때 사용한다.

# **시리즈의 행 삭제**

# 시리즈의 경우 인덱스를 한 개 또는 여러 개 지정하면 나머지로 이루어진 시리즈를 얻는다.

# In[57]:


obj = pd.Series(np.arange(5.), index=['a', 'b', 'c', 'd', 'e'])
obj


# In[58]:


new_obj = obj.drop('c')
new_obj


# In[59]:


obj.drop(['d', 'c'])


# 원본 시리즈를 직접 건드리지는 않는다.

# In[60]:


obj


# **`inplace=True` 키워드 인자**

# `inplace=True` 키워드 인자를 이용하면 원본을 수정한다.

# In[61]:


obj.drop('c', inplace=True)


# In[62]:


obj


# **데이터프레임의 행 삭제**

# 데이터프레임의 경우도 기본적으로 행의 인덱스를 기준으로 작동한다.

# In[63]:


data = pd.DataFrame(np.arange(16).reshape((4, 4)),
                    index=['Ohio', 'Colorado', 'Utah', 'New York'],
                    columns=['one', 'two', 'three', 'four'])
data


# In[64]:


data.drop(['Colorado', 'Ohio'])


# **데이터프레임의 열 삭제**

# 열을 기준으로 작동하게 하려면 `axis=1`로 지정한다.

# In[65]:


data.drop('two', axis=1)


# `axis='columns'`로 지정해도 된다.

# In[66]:


data.drop(['two', 'four'], axis='columns')


# **`inplace=True` 키워드 인자**

# `inplace=True` 키워드 인자를 사용하면 이번에도 원본을 수정함에 주의하라.

# In[67]:


data.drop('two', axis=1, inplace=True)


# In[68]:


data


# ## `fillna()` 메서드

# 결측치를 특정 값으로 채우기

# ## `reset_index()` 메서드

# 행 삭제 또는 정렬 이후 인덱스 재설정하기

# ## 연습문제

# 참고: [(실습) 데이터프레임 핵심 2부](https://colab.research.google.com/github/codingalzi/datapy/blob/master/practices/practice-pandas_3.ipynb)
