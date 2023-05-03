#!/usr/bin/env python
# coding: utf-8

# (sec:pandas_2)=
# # 데이터프레임 핵심 1부

# **주요 내용**

# `Series`와 `DataFrame` 객체를 다루는 다양한 도구를 살펴본다.
# 
# * 리인덱싱
# * 인덱싱
# * 슬라이싱

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


# ## 리인덱싱

# `reindex()` 메서드를 이용하여 시리즈의 `index`와 데이터프레임의 `index`, `columns` 속성을 임의로 재설정한다.

# ### 시리즈 리인덱싱

# In[5]:


obj = pd.Series([4.5, 7.2, -5.3, 3.6], index=['d', 'b', 'a', 'c'])
obj


# 새로운 인덱스가 추가되면 `NaN`이 사용된다.

# In[6]:


obj2 = obj.reindex(['a', 'b', 'c', 'd', 'e'])
obj2


# 지정되지 않은 인덱스는 무시된다.

# In[7]:


obj2 = obj.reindex(['a', 'c', 'd', 'e'])
obj2


# ### 데이터프레임 리인덱싱

# 데이터프레임은 (행의) `index`와 (열의) `columns` 속성에 대해 
# 리인덱싱이 가능하며 작동법은 시리지의 인덱싱과 동일하다.

# In[8]:


frame = pd.DataFrame(np.arange(9).reshape((3, 3)),
                     index=['a', 'c', 'd'],
                     columns=['Ohio', 'Texas', 'California'])
frame


# `reindex()` 메서드는 기본적으로 행의 `index` 에 대해 작동한다.

# In[9]:


frame2 = frame.reindex(['a', 'b', 'c', 'd'])
frame2


# 열의 `columns`에 대해서는 `columns` 키워드 인자를 활용한다.

# In[10]:


states = ['Texas', 'Utah', 'California']
frame.reindex(columns=states)


# ### 리인덱싱과 결측치

# 리인덱싱 과정에서 결측치가 발생할 때 여러 방식으로 채울 수 있다.

# **결측치 채우기 1: `method` 키워드 인자**

# `method='fill'` 키워드 인자는 결측치를 위쪽에 위치한 값으로 채운다.
# 
# __주의사항:__ 인덱스가 오름 또는 내림 차순으로 정렬되어 있는 경우에만 가능하다.

# In[11]:


obj3 = pd.Series(['blue', 'purple', 'yellow'], index=[0, 2, 5])
obj3


# In[12]:


obj3.reindex(range(6), method='ffill')


# 물론 위쪽에 위치한 값이 없으면 결측치가 된다.

# In[13]:


obj3.reindex(range(-1, 6), method='ffill')


# 아랫쪽에 있는 값으로 채울 수도 있다.

# In[14]:


obj3.reindex(range(-1, 6), method='bfill')


# 아니면 가장 가까운 곳에 있는 값으로 채울 수도 있다.
# 1번 인덱스의 경우처럼 거리가 같으면 아랫쪽에서 택한다.

# In[15]:


obj3.reindex(range(-1, 6), method='nearest')


# **결측치 채우기 2: `fill_value` 키워드 인자**

# 리인덱싱 과정에서 발생하는 모든 결측치를 지정된 값으로 대체할 수 있다.
# 기본값은 `NaN` 이다.

# In[16]:


obj3.reindex(range(-1, 6), fill_value='No Color')


# 리인덱싱은 항상 새로운 시리즈를 생성한다.
# 따라서 `obj3` 자체는 변하지 않는다.

# In[17]:


obj3


# ## 시리즈 인덱싱/슬라이싱

# **시리즈 인덱싱**

# 시리즈의 경우 1차원 넘파이 어레이와 거의 동일하게 작동한다.
# 다만 정수 대신에 지정된 인덱스를 사용할 때 조금 차이가 있다.

# In[18]:


obj = pd.Series(np.arange(4.), index=['a', 'b', 'c', 'd'])
obj


# 시리즈 인덱싱은 인덱스 라벨 또는 정수를 사용할 수 있다.

# In[19]:


obj['b']


# In[20]:


obj[1]


# **시리즈 슬라이싱**

# 정수를 이용한 슬라이싱은 1차원 어레이의 경우와 동일하다.

# In[21]:


obj[2:4]


# 반면에 정수가 아닌 다른 인덱스 라벨을 이용하는 슬라이싱은 양쪽 구간의 끝을 모두 
# 포함하는 점이 다르다.

# In[22]:


obj['b':'c']


# In[23]:


obj['b':'c'] = 5
obj


# __주의사항:__ 인덱스 라벨 슬라이싱은 기본적으로 알파벳 순서를 따르며
# 시리즈에 사용된 순서와 상관 없다.

# In[24]:


obj.reindex(['b', 'd', 'c', 'a'])


# In[25]:


obj['b':'d']


# **시리즈 팬시 인덱싱**

# 여러 개의 인덱스를 리스트로 지정하여 팬시 인덱싱을 진행할 수 있다.

# - 인덱스 라벨 활용

# In[26]:


obj[['b', 'a', 'd']]


# - 정수 인덱스 활용

# In[27]:


obj[[1, 3]]


# **시리즈 부울 인덱싱(필터링)**

# 부울 인덱싱(필터링)은 동일하게 작동한다.

# In[28]:


obj < 2


# In[29]:


obj[obj < 2]


# ## 데이터프레임 인덱싱

# 2차원 넘파이 어레이와 거의 유사하게 작동한다.

# In[30]:


state_dict = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada', 'NY', 'NY', 'NY'],
         'year': [2000, 2001, 2002, 2001, 2002, 2003, 2002, 2003, 2004],
         'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2, 8.3, 8.4, 8.5]}

df = pd.DataFrame(state_dict, columns=['year', 'state', 'pop', 'debt'],
                      index=['one', 'two', 'three', 'four',
                             'five', 'six', 'seven', 'eight', 'nine'])
df


# ### 열 인덱싱

# 열 인덱싱은 시리즈, 사전 등과 동일한 방식을 사용한다.
# 다만, 지정된 열의 이름을 사용한다.
# 예를 들어, `state` 열을 확인하면 시리즈로 보여준다.

# In[31]:


df['state']


# 대괄호 대신 속성 형식을 사용할 수도 있다.
# 아래 코드는 `year` 열을 시리즈로 보여준다.

# In[32]:


df.year


# **주의사항** 
# 
# 대괄호를 사용하는 인덱싱은 임의의 문자열을 사용한다.
# 반면에 속성 형식은 변수를 사용하듯이 처리한다. 
# 따라서 속성 형식에 사용될 수 있는 열의 이름은 일반 변수의 이름을 짓는 형식을 따라야 한다.
# 
# 예를 들어, Ohio 주(state)인지 여부를 판정하는 'Ohio state' 라는 열을 추가해보자.
# 아래 코드는 새로운 열을 추가하기 위해 사전의 경우처럼 대괄호를 이용하여 새로운 열의 이름과 값을 지정한다.

# In[33]:


df['Ohio state'] = df.state == 'Ohio'
df


# 그러면 `'Ohio state'`의 열을 확인하는 방법은 대괄호만 이용할 수 있으며 속성 형식은 불가능하다.

# In[34]:


df['Ohio state']


# 아래와 같이 실행하면 문법 오류(`SyntaxError`)가 발생한다.
# 이유는 `Ohio state`가 변수 이름으로 허용되지 않기 때문이다.
# 
# ```python
# >>> df.Ohio state
#   Cell In [122], line 1
#     df.Ohio state
#             ^
# SyntaxError: invalid syntax
# ```

# **열 팬시 인덱싱**

# 열 인덱스의 리스트를 사용하면 팬시 인덱싱처럼 작동하고 데이터프레임을 생성한다.

# In[35]:


df[['state', 'state', 'year', 'year', 'year']]


# 길이가 1인 리스트일 때도 데이터프레임을 생성한다.

# In[36]:


df[['state']]


# **열 업데이트**

# 열 인덱싱을 이용하여 항목의 값을 지정할 수 있다. 
# 아래 코드는 `'debt'` 열의 값을 16.5로 일정하게 지정한다.
# 
# __참고:__ 브로드캐스팅이 기본적으로 작동한다.

# In[37]:


df['debt'] = 16.5
df


# 반면에 행의 길이와 동일한 리스트, 어레이 등을 이용하여 각 행별로 다른 값을 지정할 수 있다.
# 리스트, 어레이의 길이가 행의 개수와 동일해야 함에 주의해야 한다.

# In[38]:


df['debt'] = np.arange(9.)
df


# 반면에 시리즈를 이용하여 특정 열의 값을 지정할 수 있으며, 이 때는 항목의 길이가 
# 행의 개수와 동일할 필요가 없다.
# 다만, 지정된 행의 인덱스 값만 삽입되며 나머지는 `NaN`이 삽입된다.

# In[39]:


val = pd.Series([-1.2, -1.5, -1.7, 2.2], index=['two', 'four', 'five', 'eleven'])
val


# 위 시리즈를 이용하여 `'debt'` 열의 값을 업데이트하면 다음과 같다.
# 
# - `'two'`, `'four'`, `'five'`  행은 지정된 값으로 업데이트
# - 나머지 인덱스의 값은 결측치로 처리됨.
# - `'eleven'`에 해당하는 값은 무시됨. 이유는 `df`의 인덱스로 포함되지 않기 때문임.

# In[40]:


df['debt'] = val
df


# **부울 인덱싱(필터링)**

# In[41]:


data = pd.DataFrame(np.arange(16).reshape((4, 4)),
                    index=['Ohio', 'Colorado', 'Utah', 'New York'],
                    columns=['one', 'two', 'three', 'four'])
data


# 필터링(부울 인덱싱) 또한 넘파이 2차원 어레이와 동일하게 작동한다.

# In[42]:


mask1 = data['three'] > 5
mask1


# In[43]:


data[mask1]


# 필터링을 이용하여 특정 항목의 값을 업데이트할 수도 있다.
# 
# - `~mask1`은 `mask1`의 부정을 나타낸다.

# In[44]:


data[~mask1] = 0
data


# 각각의 항목에 대한 필터링도 비슷하게 작동한다.

# In[45]:


mask2 = data < 6
mask2


# In[46]:


data[mask2] = 0
data


# ### 행 인덱싱

# **`loc()`와 `iloc()` 함수**

# `loc()` 또는 `iloc()` 함수를 이용한다.
# 
# - `loc()` 함수: 인덱스 라벨을 이용할 경우
# - `iloc()` 함수: 정수 인덱스를 이용할 경우

# In[47]:


data.loc['Colorado']


# In[48]:


data.iloc[2]


# **행 팬시 인덱싱**

# 여러 행을 대상으로 인덱싱 하려면 아래와 같이 인덱스 라벨의 리스트 또는 정수의 리스트를 활용한다.
# 결과로 데이터프레임이 생성된다.

# - 인덱스 라벨 리스트 활용

# In[49]:


data.loc[['New York', 'Colorado', 'Colorado', 'Utah']]


# - 정수 리스트 활용

# In[50]:


data.iloc[[3, 1, 1, 2]]


# ## 데이터프레임 슬라이싱: `loc()`과 `iloc()` 함수

# `loc()` 과 `iloc()` 함수를 행과 열에 동시에 적용할 수 있으며,
# 2차원 넘파이 어레이에 대한 인덱싱/슬라이싱이 작동하는 방식과 거의 동일하다.

# **열 슬라이싱**

# 열에 대해서만 슬라이싱을 적용하려면 행을 전체로 지정한다.

# - 행: `[:]`
# - 열: `[:3]`

# In[51]:


data.iloc[:, :3]


# **행 슬라이싱**

# 행에 대해서만 슬라이싱을 적용하려면 열을 전체로 지정한다.

# - 행: `[:]`
# - 열: `[:3]`

# In[52]:


data.iloc[2:, :]


# 열 슬라이싱은 지정하지 않아도 된다.

# In[53]:


data.iloc[2:]


# **스텝 활용**

# 스텝(step)도 사용할 수 있다.

# In[54]:


data.iloc[::2, 1::2]


# **인덱싱과 슬라이싱 조합**

# 인덱싱과 슬라이싱의 어떤 조합도 행과 열에 대해 적용할 수 있다.

# - 행: `Colorado`
# - 열: `['two', 'three']`

# In[55]:


data.loc['Colorado', ['two', 'three']]


# - 행: `2`
# - 열: `[3, 0, 1]`

# In[56]:


data.iloc[2, [3, 0, 1]]


# - 행: `[1, 2]`
# - 열: `[3, 0, 1]`

# In[57]:


data.iloc[[1, 2], [3, 0, 1]]


# - 행: `[:Colorado]`
# - 열: `'two'`

# In[58]:


data.loc[:'Colorado', 'two']


# - 행: `[:Colorado]`
# - 열: `['two':]`

# In[59]:


data.loc[:'Colorado', 'two':]


# **부울 인덱싱(필터링) 연속 적용**

# 인덱싱/슬라이싱은 시리즈 또는 데이터프레임을 생성하기에 바로 이어 필터링을 연달아 적용하면
# 보다 간편하게 원하는 시리즈 또는 데이터프레임을 생성할 수 있다.

# In[60]:


data.iloc[:, :3][data.three > 5]


# ## 연습문제

# 참고: [(실습) 데이터프레임 핵심 1부](https://colab.research.google.com/github/codingalzi/datapy/blob/master/practices/practice-pandas_2.ipynb)
