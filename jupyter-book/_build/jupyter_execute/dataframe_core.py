#!/usr/bin/env python
# coding: utf-8

# # 데이터프레임 핵심

# **주요 내용**

# `Series`와 `DataFrame` 객체를 다루는 다양한 도구를 살펴본다.
# 
# * 리인덱싱
# * 삭제
# * 연산
# * 정렬

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


PREVIOUS_MAX_ROWS = pd.options.display.max_rows # 원래 60이 기본.
pd.set_option("max_rows", 20)


# ## 리인덱싱

# ### 시리즈 리인덱싱

# In[4]:


obj = pd.Series([4.5, 7.2, -5.3, 3.6], index=['d', 'b', 'a', 'c'])
obj


# 새로운 인덱스가 추가되면 `NaN`이 사용된다.

# In[5]:


obj2 = obj.reindex(['a', 'b', 'c', 'd', 'e'])
obj2


# 지정되지 않은 인덱스는 무시된다.

# In[6]:


obj2 = obj.reindex(['a', 'c', 'd', 'e'])
obj2


# ### 데이터프레임 리인덱싱

# 데이터프레임은 (행의) `index`와 (열의) `columns` 속성에 대해 
# 리인덱싱이 가능하며 작동법은 시리지의 인덱싱과 동일하다.

# In[7]:


frame = pd.DataFrame(np.arange(9).reshape((3, 3)),
                     index=['a', 'c', 'd'],
                     columns=['Ohio', 'Texas', 'California'])
frame


# `reindex()` 메서드는 기본적으로 행의 `index` 에 대해 작동한다.

# In[8]:


frame2 = frame.reindex(['a', 'b', 'c', 'd'])
frame2


# 열의 `columns`에 대해서는 `columns` 키워드 인자를 활용한다.

# In[9]:


states = ['Texas', 'Utah', 'California']
frame.reindex(columns=states)


# `reindex()` 메서드와 함께 사용할 수 있는 키워드 인자가 더 있지만 여기서는 다루지 않는다.

# ## 결측치 처리

# **결측치 채우기 1: `method` 키워드 인자**

# 리인덱싱 과정에서 결측치가 발생할 때 여러 방식으로 채울 수 있다.
# `method='fill'` 키워드 인자는 결측치를 위쪽에 위치한 값으로 채운다.
# 
# __주의사항:__ 인덱스가 오름 또는 내림 차순으로 정렬되어 있는 경우에만 가능하다.

# In[10]:


obj3 = pd.Series(['blue', 'purple', 'yellow'], index=[0, 2, 5])
obj3


# In[11]:


obj3.reindex(range(6), method='ffill')


# 물론 위쪽에 위치한 값이 없으면 결측치가 된다.

# In[12]:


obj3.reindex(range(-1, 6), method='ffill')


# 아랫쪽에 있는 값으로 채울 수도 있다.

# In[13]:


obj3.reindex(range(-1, 6), method='bfill')


# 아니면 가장 가까운 곳에 있는 값으로 채울 수도 있다.
# 1번 인덱스의 경우처럼 거리가 같으면 아랫쪽에서 택한다.

# In[14]:


obj3.reindex(range(-1, 6), method='nearest')


# **결측치 채우기 2: `fill_value` 키워드 인자**

# 리인덱싱 과정에서 발생하는 모든 결측치를 지정된 값으로 대체할 수 있다.
# 기본값은 `NaN` 이다.

# In[15]:


obj3.reindex(range(-1, 6), fill_value='No Color')


# 리인덱싱은 항상 새로운 시리즈를 생성한다.
# 따라서 `obj3` 자체는 변하지 않는다.

# In[16]:


obj3


# ## 인덱싱, 슬라이싱, 필터링

# ### 시리즈의 인덱싱, 슬라이싱, 필터링

# 시리즈의 경우 1차원 넘파이 어레이와 거의 동일하게 작동한다.
# 다만 정수 대신에 지정된 인덱스를 사용할 때 조금 차이가 있다.

# In[17]:


obj = pd.Series(np.arange(4.), index=['a', 'b', 'c', 'd'])
obj


# In[18]:


obj['b']


# In[19]:


obj[1]


# In[20]:


obj[2:4]


# 여러 개의 인덱스를 리스트로 지정하여 인덱싱을 진행할 수 있다.

# In[21]:


obj[['b', 'a', 'd']]


# In[22]:


obj[[1, 3]]


# 필터링(부울 인덱싱)은 동일하게 작동한다.

# In[23]:


obj < 2


# In[24]:


obj[obj < 2]


# 정수가 아닌 다른 인덱스 이름, 즉 라벨을 이용하는 슬라이싱은 양쪽 구간의 끝을 모두 
# 포함하는 점이 다르다.

# In[25]:


obj['b':'c']


# In[26]:


obj['b':'c'] = 5
obj


# __주의사항:__ 라벨 슬라이싱은 기본적으로 알파벳 순서를 따르며
# 시리즈에 사용된 순서와 상관 없다.

# In[27]:


obj.reindex(['b', 'd', 'c', 'a'])


# In[28]:


obj['b':'d']


# ### 데이터프레임의 인덱싱, 슬라이싱, 필터링

# 2차원 넘파이 어레이와 거의 유사하게 작동한다.

# In[29]:


data = pd.DataFrame(np.arange(16).reshape((4, 4)),
                    index=['Ohio', 'Colorado', 'Utah', 'New York'],
                    columns=['one', 'two', 'three', 'four'])
data


# 인덱싱인 기본적으로 열을 기준으로 진행된다.

# In[30]:


data['two']


# In[31]:


data[['three', 'one']]


# 하지만 숫자 슬라이싱은 행을 기준으로 작동한다.

# In[32]:


data[:2]


# 필터링(부울 인덱싱) 또한 넘파이 2차원 어레이와 동일하게 작동한다.

# In[33]:


mask1 = data['three'] > 5
mask1


# In[34]:


data[mask1]


# 필터링을 이용하여 특정 항목의 값을 업데이트할 수도 있다.
# 
# - `~mask1`은 `mask1`의 부정을 나타낸다.

# In[35]:


data[~mask1] = 0
data


# 각각의 항목에 대한 필터링도 비슷하게 작동한다.

# In[36]:


mask2 = data < 6
mask2


# In[37]:


data[mask2] = 0
data


# **행 단위 인덱싱/슬라이싱**

# `loc()` 또는 `iloc()` 메서드를 이용한다.
# 
# - `loc()` 메서드: 라벨을 이용할 경우
# - `iloc()` 메서드: 정수 인덱스를 이용할 경우

# In[38]:


data.loc['Colorado']


# In[39]:


data.iloc[2]


# 행과 열에 대해 동시에 인덱싱/슬라이싱을 사용할 수 있으며
# 2차원 넘파이 어레이가 작동하는 방식과 비슷하다.

# In[40]:


data.loc['Colorado', ['two', 'three']]


# In[41]:


data.iloc[2, [3, 0, 1]]


# In[42]:


data.iloc[[1, 2], [3, 0, 1]]


# In[43]:


data.loc[:'Colorado', 'two']


# 인덱싱/슬라이싱에 이은 필터링을 연달아 적용할 수도 있다.

# In[44]:


data.iloc[:, :3][data.three > 5]


# ### `drop()` 메서드

# 특정 행 또는 열의 인덱스를 제외한 나머지로 이루어진 시리즈/데이터프레임을 구할 때 사용한다.
# 
# 시리즈의 경우 인덱스를 한 개 또는 여러 개 지정하면 나머지로 이루어진 시리즈를 얻는다.

# In[45]:


obj = pd.Series(np.arange(5.), index=['a', 'b', 'c', 'd', 'e'])
obj


# In[46]:


new_obj = obj.drop('c')
new_obj


# In[47]:


obj.drop(['d', 'c'])


# 원래의 시리즈를 직접 건드리지는 않는다.

# In[48]:


obj


# 하지만 `inplace=True` 키워드 인자를 이용하여 원본을 수정할 수도 있다.
# 물론 사용에 매우 주의해야 한다.

# In[49]:


obj.drop('c', inplace=True)


# In[50]:


obj


# 데이터프레임의 경우도 기본적으로 행의 인덱스를 기준으로 작동한다.

# In[51]:


data = pd.DataFrame(np.arange(16).reshape((4, 4)),
                    index=['Ohio', 'Colorado', 'Utah', 'New York'],
                    columns=['one', 'two', 'three', 'four'])
data


# In[52]:


data.drop(['Colorado', 'Ohio'])


# 열을 기준으로 작동하게 하려면 `axis=1`로 지정한다.

# In[53]:


data.drop('two', axis=1)


# `axis='columns'`로 지정해도 된다.

# In[54]:


data.drop(['two', 'four'], axis='columns')


# `inplace=True` 키워드 인자를 사용하면 이번에도 원본을 수정함에 주의하라.

# In[55]:


data.drop('two', axis=1, inplace=True)


# In[56]:


data


# ## 산술 연산

# 시리즈/데이터프레임의 사칙 연산은 기본적으로 아래 원칙을 따른다.
# 
# - 연산에 사용된 모든 인덱스는 포함
# - 공통으로 사용되는 인덱스의 항목에 대해서만 연산 적용. 그렇지 않으면 `NaN`으로 처리.

# In[57]:


s1 = pd.Series([7.3, -2.5, 3.4, 1.5], index=['a', 'c', 'd', 'e'])
s1


# In[58]:


s2 = pd.Series([-2.1, 3.6, -1.5, 4, 3.1],
               index=['a', 'c', 'e', 'f', 'g'])
s2


# In[59]:


s1 + s2


# In[60]:


df1 = pd.DataFrame(np.arange(9.).reshape((3, 3)), 
                   columns=list('bcd'),
                   index=['Ohio', 'Texas', 'Colorado'])
df1


# In[61]:


df2 = pd.DataFrame(np.arange(12.).reshape((4, 3)), columns=list('bde'),
                   index=['Utah', 'Ohio', 'Texas', 'Oregon'])
df2


# In[62]:


df1 + df2


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

# ### 연산과 결측치

# 공통 인덱스가 아니거나 결측치가 이미 존재하는 경우 기본적으로 결측치로 처리된다.
# 하지만 `fill_value` 키워드 인자를 이용하여 지정된 값으로 처리하게 만들 수도 있다.
# 다만, 연산 기호 대신에 해당 연산의 메서드를 활용해야 한다.

# In[63]:


df1.add(df2, fill_value=0)


# ### 브로드캐스팅

# 넘파이에서 2차원 어레이와 1차원 어레이 사이에
# 브로드캐스팅이 가능한 경우,
# 즉, 차원을 맞출 수 있는 경우에 연산이 가능했다.

# In[64]:


arr = np.arange(12.).reshape((3, 4))
arr


# In[65]:


arr[0]


# In[66]:


arr - arr[0]


# 브로드캐스팅이 불가능하면 오류가 발생한다.

# In[67]:


arr[:,1]


# In[68]:


try:
    arr + arr[:, 1]
except:
    print("브로드캐스팅 불가능!")


# 물론 아래와 같이 브로드캐스팅이 가능하도록 모양을 변환한 다음엔 연산이 가능하다.

# In[69]:


arr_1 = arr[:,1][:, np.newaxis]
arr_1


# In[70]:


arr + arr_1


# 데이터프레임과 시리즈 사이의 연산도 동일하게 작동한다.
# 다만, 행 또는 열에 대한 연산 여부를 확실하게 구분해주어야 한다.

# In[71]:


frame = pd.DataFrame(np.arange(12.).reshape((4, 3)),
                     columns=list('bde'),
                     index=['Utah', 'Ohio', 'Texas', 'Oregon'])
frame


# In[72]:


series = frame.iloc[0]
series


# 브로드캐스팅은 기본족으로 행 단위로 이루어진다.
# 따라서 아래처럼 데이터프레임과 시리즈의 연산을 그냥 적용할 수 있다.

# In[73]:


frame - series


# 공통 인덱스가 존재하면 두 인자 모두에 대해 브로드캐스팅이 적용된다.

# In[74]:


series2 = pd.Series(range(3), index=['b', 'e', 'f'])
series2


# In[75]:


frame + series2


# 열 단위로 데이터프레임과 시리즈를 더하려면 
# 해당 연산 메서드를 `axis=0` 키워드 인자와 함께 적용해야 한다.

# In[76]:


series3 = frame['d']
series3


# In[77]:


frame.sub(series3, axis=0)


# `axis='index'`를 사용해도 된다.

# In[78]:


frame.sub(series3, axis='index')


# ### 유니버설 함수

# 유니버설 함수는 넘파이의 경우와 동일하게 작동한다.

# In[79]:


frame = pd.DataFrame(np.random.randn(4, 3), columns=list('bde'),
                     index=['Utah', 'Ohio', 'Texas', 'Oregon'])
frame


# 넘파이의 `abs()` 함수를 적용하면 항목별로 이루어진다.

# In[80]:


np.abs(frame)


# 시리즈에 대해서도 동일하다.

# In[81]:


np.abs(frame['b'])


# **`map()`과 `applymap()` 메서드**

# 유니버설 함수가 아닌 함수를 시리즈의 항목별로 적용하려면
# `map()` 메서드를 이용한다.
# 
# 예를 들어 아래 람다(lambda) 함수는 부동소수점을 소수점 이하 셋째 자리에서 반올림한 값만 보여주도록 
# 한다.

# In[82]:


format = lambda x: '%.2f' % x


# 시리즈에 적용해보자.

# In[83]:


frame['e'].map(format)


# 유니버설 함수가 아닌 함수를 데이터프레임의 항목별로 적용하려면
# `applymap()` 메서드를 이용한다.

# In[84]:


frame.applymap(format)


# **`apply()` 메서드**

# 행 또는 열 단위로 함수를 적용하려면 `apply()` 메서드를 활용한다.
# 기본은 열 단위로 함수가 적용되며 반환값이 스칼라 값이면 시리즈가 반환된다.
# 
# 예를 들어 아래 함수는 최댓값과 최소값의 차이를 반환한다.

# In[85]:


f1 = lambda x: x.max() - x.min()


# 데이터프레임에 적용하면 열 별로 최댓값과 최솟값의 차이를 계산하여 시리즈로 반환한다.

# In[86]:


frame.apply(f1)


# 행 별로 함수를 적용하려면 `axis=1` 또는 `axis='columls'`를 지정해야 한다.

# In[87]:


frame.apply(f1, axis='columns')


# In[88]:


frame.apply(f1, axis=1)


# 함수의 반환값이 시리즈이면 `apply()` 메서드는 데이터프레임을 반환된다.
# 예를 들어 아래 함수는 최솟값과 최댓값을 갖는 시리즈를 반환한다.

# In[89]:


def f2(x):
    return pd.Series([x.min(), x.max()], index=['min', 'max'])


# `apply()` 메서드와 함께 호출하면 열 별로 최댓값과 최솟값을 계산하여 데이터프레임으로 반환한다.

# In[90]:


frame.apply(f2)


# __참고:__ 시리즈 객체 또한 `apply()` 메서드를 갖는다. 하지만 
# 기본적으로 `map()` 메서드처럼 작동한다. 
# `map()` 메서드보다 좀 더 다야한 기능을 갖지만 여기서는 다루지 않는다.

# ## 정렬

# 행과 열의 인덱스 또는 항목을 대상으로 정렬할 수 있다.

# **`sort_index()` 메서드**

# 시리즈의 경우 인덱스를 기준으로 정렬한다. 

# In[91]:


obj = pd.Series(range(4), index=['d', 'a', 'b', 'c'])
obj


# In[92]:


obj.sort_index()


# 내림차순으로 정렬하려면 `ascending=False` 키워드 인자를 함께 사용한다.

# In[93]:


obj.sort_index(ascending=False)


# 데이터프레임의 경우 행 또는 열의 인덱스를 기준으로 정렬한다. 

# In[94]:


frame = pd.DataFrame(np.arange(8).reshape((2, 4)),
                     index=['three', 'one'],
                     columns=['d', 'a', 'b', 'c'])
frame


# 기본은 행의 인데스를 기준으로 정렬한다.

# In[95]:


frame.sort_index()


# 열의 인덱스를 기준으로 정렬하려면 `axis=1` 또는 `axis='columns'` 키워드 인자를 사용한다.

# In[96]:


frame.sort_index(axis=1)


# In[97]:


frame.sort_index(axis='columns')


# 내림차순으로 정렬하려면 `ascending=False` 키워드 인자를 함께 사용한다.

# In[98]:


frame.sort_index(axis=1, ascending=False)


# **`sort_values()` 메서드**

# 지정된 열 또는 행에 속한 값들을 기준으로 정렬할 때 사용한다.

# In[99]:


obj = pd.Series([4, 7, -3, 2])
obj


# In[100]:


obj.sort_values()


# 결측치는 맨 나중에 위치시킨다.

# In[101]:


obj = pd.Series([4, np.nan, 7, np.nan, -3, 2])
obj


# In[102]:


obj.sort_values()


# 데이터프레임의 경우 `by` 키워드 인자를 이용하여 열의 라벨을 지정해야 한다.

# In[103]:


frame = pd.DataFrame({'b': [4, 7, -3, 2], 'a': [0, 1, 0, 1]})
frame


# 예를 들어 `b` 열의 값을 기준으로 정렬한다.
# 물론 동일한 행의 값은 함께 움직인다.

# In[104]:


frame.sort_values(by='b')


# 여러 열의 값을 기준으로 정렬하려면 라벨의 리스트를 입력한다.
# 그러면 리스트 항목 순서대로 기준이 정해진다.
# 
# 예를 들어 아래 코드는 먼저 `a` 열의 항목들을 순서대로 정렬한 다음에
# 동등한 값의 경우에는 `b` 열의 항목들 순서대로 정렬한다.

# In[105]:


frame.sort_values(by=['a', 'b'])


# __참고:__ `axis=1`을 이용하여 특정 행의 값을 기준으로 정렬할 수도 있다.
# 하지만 데이터프레임은 서로 다른 특성을 열 단위로 담는 목적으로
# 사용되기에 이런 정렬은 사용할 이유가 별로 없다.

# In[106]:


frame.sort_values(by=2, axis=1, ascending=False)


# ## 연습문제

# 참고: [(실습) 데이터프레임 핵심](https://colab.research.google.com/github/codingalzi/datapy/blob/master/practices/practice-dataframe_core.ipynb)
