#!/usr/bin/env python
# coding: utf-8

# (sec:pandas10min_2)=
# # 판다스 10분 완성 2부
# 

# **필수 라이브러리**

# In[1]:


import numpy as np
import pandas as pd


# ## 합병과 결합: merge-join-concat

# - 참고: [Merging section](https://pandas.pydata.org/docs/user_guide/merging.html#merging)

# ### 종/횡 결합: `pd.concat()` 함수

# `pd.concat()` 함수는 여러 개의 데이터프레임을 하나로 합친다.
# 
# - `axis=0`: 종 결합. 즉 데이터프레임 여러 개의 위아래 결합.

# In[2]:


df1 = pd.DataFrame(
    {
        "A": ["A0", "A1", "A2", "A3"],
        "B": ["B0", "B1", "B2", "B3"],
        "C": ["C0", "C1", "C2", "C3"],
        "D": ["D0", "D1", "D2", "D3"],
    },
    index=[0, 1, 2, 3],
)

df1


# In[3]:


df2 = pd.DataFrame(
    {
        "A": ["A4", "A5", "A6", "A7"],
        "B": ["B4", "B5", "B6", "B7"],
        "C": ["C4", "C5", "C6", "C7"],
        "D": ["D4", "D5", "D6", "D7"],
    },
    index=[4, 5, 6, 7],
)

df2


# In[4]:


df3 = pd.DataFrame(
    {
        "A": ["A8", "A9", "A10", "A11"],
        "B": ["B8", "B9", "B10", "B11"],
        "C": ["C8", "C9", "C10", "C11"],
        "D": ["D8", "D9", "D10", "D11"],
    },
    index=[8, 9, 10, 11],
)

df3


# In[5]:


pd.concat([df1, df2, df3]) # axis=0 이 기본값


# - `axis=1`: 횡 결합. 즉 데이터프레임 여러 개의 좌우 결합.

# In[6]:


df4 = pd.DataFrame(
    {
        "B": ["B2", "B3", "B6", "B7"],
        "D": ["D2", "D3", "D6", "D7"],
        "F": ["F2", "F3", "F6", "F7"],
    },
    index=[2, 3, 6, 7],
)

df4


# In[7]:


pd.concat([df1, df4], axis=1)


# 인덱스를 기존의 데이터프레임과 통일시키기 위해 리인덱싱을 활용할 수도 있다.

# In[8]:


df1.index


# In[9]:


pd.concat([df1, df4], axis=1).reindex(df1.index)


# In[10]:


pd.concat([df1, df4.reindex(df1.index)], axis=1)


# ### 합병: `pd.merge()` 함수

# `pd.merge()` 함수 는 SQL 방식으로 특정 열을 기준으로 두 개의 데이터프레임을 합친다.
# 다양한 옵션을 지원하는 매우 강력한 도구이다.
# 
# - 참고: [Database style joining](https://pandas.pydata.org/docs/user_guide/merging.html#merging-join)

# **예제**

# 실습을 위해 아래 두 데이터프레임을 이용한다.

# In[11]:


left = pd.DataFrame({"key": ["foo", "foo"], "lval": [1, 2]})
right = pd.DataFrame({"key": ["foo", "foo"], "rval": [4, 5]})


# In[12]:


left


# In[13]:


right


# - `on="key"` 키워드 인자
#     - `key` 열에 사용된 항목 각각에 대해 다른 열에서 해당 항목과 연관된 값들을 조합할 수 있는 모든 경우의 수를 다룬다.
#     - `foo` 값에 대해 `lval` 열에서 2개의 값이,
#         `rval` 열에서 2개의 값이 있기에 `foo`와 관련해서 총 4개의 경우가 생성된다.
#     <br><br>
#     
#     | `key` | `left.lval` | `right.rval` | 경우의 수 |
#     | :---: | :---: | :---: | :---: |
#     | `foo` | `1, 2` | `4, 5` | 4 |

# In[14]:


pd.merge(left, right, on="key")


# **예제**

# In[15]:


left = pd.DataFrame({"key": ["foo", "bar"], "lval": [1, 2]})
right = pd.DataFrame({"key": ["foo", "bar"], "rval": [4, 5]})


# In[16]:


left


# In[17]:


right


# - `on="key"` 키워드 인자
#     - `key` 열에 사용된 항목별로 모든 경우의 수를 다룬다.
#     - `foo` 값에 대해 `lval` 열에서 1개의 값이,
#         `rval` 열에서 1개의 값이 있기에 `foo`와 관련해서 총 1개의 경우가 생성된다.
#     - `bar` 값에 대해 `lval` 열에서 1개의 값이,
#         `rval` 열에서 1개의 값이 있기에 `foo`와 관련해서 총 1개의 경우가 생성된다.
#     <br><br>
#     
#     | `key` | `left.lval` | `right.rval` | 경우의 수 |
#     | :---: | :---: | :---: | :---: |
#     | `foo` | `1` | `4` | 1 |        
#     | `bar` | `2` | `5` | 1 |        

# In[18]:


pd.merge(left, right, on="key")


# **예제**

# 경우의 수는 지정된 열의 항목이 사용된 횟수를 기준으로 한다. 

# In[19]:


left = pd.DataFrame(
    {
        "key": ["K0", "K1", "K2", "K3"],
        "A": ["A0", "A1", "A2", "A3"],
        "B": ["B0", "B1", "B2", "B3"],
    }
)


# In[20]:


left


# In[21]:


right = pd.DataFrame(
    {
        "key": ["K0", "K1", "K2", "K3"],
        "C": ["C0", "C1", "C2", "C3"],
        "D": ["D0", "D1", "D2", "D3"],
    }
)


# In[22]:


right


# | `key` | (`left.A`, `left.B`) | (`right.C`, `right.D`) | 경우의 수 |
# | :---: | :---: | :---: | :---: |
# | `K0` | (`A0`, `B0`) | (`C0`, `D0`) | 1 |
# | `K1` | (`A1`, `B1`) | (`C1`, `D1`) | 1 |
# | `K2` | (`A2`, `B2`) | (`C2`, `D2`) | 1 |
# | `K3` | (`A3`, `B3`) | (`C3`, `D3`) | 1 |

# In[23]:


result = pd.merge(left, right, on="key")
result


# **다양한 키 활용**
# 
# - 두 개 이상의 키를 하나의 쌍으로 된 키를 사용하는 경우와 유사함. 

# In[24]:


left = pd.DataFrame(
    {
        "key1": ["K0", "K0", "K1", "K2"],
        "key2": ["K0", "K1", "K0", "K1"],
        "A": ["A0", "A1", "A2", "A3"],
        "B": ["B0", "B1", "B2", "B3"],
    }
)

left


# In[25]:


right = pd.DataFrame(
    {
        "key1": ["K0", "K1", "K1", "K2"],
        "key2": ["K0", "K0", "K0", "K0"],
        "C": ["C0", "C1", "C2", "C3"],
        "D": ["D0", "D1", "D2", "D3"],
    }
)

right


# - `how='inner'`: 지정된 키의 교집합 대상

# In[26]:


result = pd.merge(left, right, on=["key1", "key2"]) # how='inner' 가 기본값
result


# In[27]:


result = pd.merge(left, right, how="inner", on=["key1", "key2"])
result


# - `how='outer'`: 지정된 키의 합집합 대상

# In[28]:


result = pd.merge(left, right, how="outer", on=["key1", "key2"])
result


# - `how='left'`: 왼쪽 데이터프레임의 키에 포함된 항목만 대상

# In[29]:


left


# In[30]:


result = pd.merge(left, right, how="left", on=["key1", "key2"])
result


# - `how='right'`: 오른쪽 데이터프레임의 키에 포함된 항목만 대상

# In[31]:


right


# In[32]:


result = pd.merge(left, right, how="right", on=["key1", "key2"])
result


# - `how='cross'`: 모든 경우의 수 조합

# In[33]:


result = pd.merge(left, right, how="cross")
result


# ### 합병: `DataFrame.join()` 메서드

# 인덱스를 기준으로 두 개의 데이터프레임을 합병할 때 사용한다.

# In[34]:


left = pd.DataFrame(
    {"A": ["A0", "A1", "A2"], "B": ["B0", "B1", "B2"]}, index=["K0", "K1", "K2"]
)

left


# In[35]:


right = pd.DataFrame(
    {"C": ["C0", "C2", "C3"], "D": ["D0", "D2", "D3"]}, index=["K0", "K2", "K3"]
)

right


# In[36]:


left.join(right)


# 아래와 같이 `pd.merge()` 함수를 이용한 결과와 동일하다.

# In[37]:


pd.merge(left, right, left_index=True, right_index=True, how='left')


# `pd.merge()` 함수의 키워드 인자를 동일하게 사용할 수 있다.

# - `how='outer'`

# In[38]:


left.join(right, how="outer")


# 아래 코드가 동일한 결과를 낸다.

# In[39]:


pd.merge(left, right, left_index=True, right_index=True, how='outer')


# - `how='inner'`

# In[40]:


left.join(right, how="inner")


# 아래 코드가 동일한 결과를 낸다.

# In[41]:


pd.merge(left, right, left_index=True, right_index=True, how='inner')


# ## 다중 인덱스<font size='2'>MultiIndex</font>

# - 참고: [Multiindex / advanced indexing](https://pandas.pydata.org/docs/user_guide/advanced.html)

# 다중 인덱스를 이용하여 데이터를 보다 체계적으로 다를 수 있다.
# 또한 이어서 다룰 그룹 분류<font size='2'>Group by</font>, 
# 모양 변환<font size='2'>reshaping</font>, 
# 피벗 변환<font size='2'>pivoting</font> 등에서 유용하게 활용된다.

# ### `MultiIndex` 객체

# 다중 인덱스 객체는 보통 튜플을 이용한다.
# 예를 들어 아래 두 개의 리스트를 이용하여 튜플을 생성한 다음 다중 인덱스로 만들어보자.

# In[42]:


arrays = [
    ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
    ["one", "two", "one", "two", "one", "two", "one", "two"],
]


# - 튜플 생성: 항목 8개

# In[43]:


tuples = list(zip(*arrays))
tuples


# **다중 인덱스 객체 생성: `from_tupes()` 함수**

# 튜플 리스트를 이용하여 다중 인덱스 객체를 생성할 수 있다.

# In[44]:


index = pd.MultiIndex.from_tuples(tuples)
index


# - `names` 키워드 인자
#     - 다중 인덱스의 각 레벨<font size='2'>level</font>의 이름 지정. 
#     - 지정되지 않으면 `None`으로 처리됨.
# 
# 예를 들어 위 코드에서 사용된 각각의 레벨에 이름은 다음과 같다.
# 
# - `"first"`: 0-레벨 이름
# - `"second"`: 1-레벨 이름

# In[45]:


index = pd.MultiIndex.from_tuples(tuples, names=["first", "second"])
index


# **다중 인덱스 객체 생성: `from_arrays()` 함수**

# 길이가 동일한 여러 개의 리스트로 구성된 어레이를 직접 이용할 수도 있다.

# In[46]:


index = pd.MultiIndex.from_arrays(arrays, names=["first", "second"])
index


# ### 다중 인덱스 라벨<font size='2'>label</font>을 사용하는 시리즈/데이터프레임 객체

# - 시리즈 생성

# 아래 코드는 길이가 8인 어레이를 이용하여 시리즈를 생성한다.
# 인덱스의 라벨은 다중 인덱스가 사용된다.
# 각각의 레벨에서 라벨이 연속적으로 사용되는 경우는 보다 자연스러운 표현을 위해 생략되기도 한다.

# In[47]:


s = pd.Series(np.random.randn(8), index=index)
s


# - 데이터프레임 생성

# 아래 코드는 8개의 행으로 이뤄진 2차원 어레이를 이용하여 데이터프레임을 생성한다.
# `index` 또는 `columns`로 여러 개의 리스트로 구성된 어레이를 지정하면
# 자동으로 다중 인덱스 라벨이 지정된다.

# In[48]:


df = pd.DataFrame(np.random.randn(8, 4), index=arrays)
df


# 다중 인덱스를 열 라벨로도 활용할 수 있다.
# 아래 코드는 8개의 열로 이뤄진 2차원 어레이를 이용하여 데이터프레임을 생성한다.

# In[49]:


df1 = pd.DataFrame(np.random.randn(3, 8), index=["A", "B", "C"], columns=index)
df1


# 인덱스 라벨과 열 라벨 모두 다중 인덱스를 이용할 수도 있다.
# 
# - 동일한 길이의 리스트로 이루어진 리스트를 인덱스 또는 열의 라벨로 지정하면
#     다중 인덱스로 자동 지정된다.

# In[50]:


arrays2 = [
    ["toto", "toto", "titi", "titi", "tata", "tata"],
    ["A", "B", "A", "B", "A", "B"],
]


# In[51]:


pd.DataFrame(np.random.randn(6, 6), index=index[:6], columns=arrays2)


# **주의사항**

# 튜플을 라벨로 사용하는 것은 다중 인덱스와 아무 상관 없다.
# 단지 라벨이 튜플인 것 뿐이다.

# In[52]:


tuples


# In[53]:


pd.Series(np.random.randn(8), index=tuples)


# ### 인덱스의 레벨

# 다중 인덱스 객체의 `get_level_values()` 메서드를 이용하여 레벨별 인덱스 라벨을 확인할 수 있다.

# - 0-레블 라벨

# In[54]:


index.get_level_values(0)


# 레벨 이름을 이용할 수도 있다.

# In[55]:


index.get_level_values("first")


# - 1-레블 라벨

# In[56]:


index.get_level_values(1)


# In[57]:


index.get_level_values("second")


# ### 인덱싱

# 다중 인덱스를 라벨로 사용하는 시리즈와 데이터프레임의 인덱싱은 일반 인덱싱과 크게 다르지 않다.

# - 시리즈 인덱싱

# In[58]:


s


# In[59]:


s["qux"]


# - 데이터프레임 인덱싱

# In[60]:


df


# In[61]:


df.loc["bar"]


# 레벨별로 라벨을 지정할 수 있다. 각각의 라벨은 쉼표로 구분한다.

# In[62]:


df.loc["bar", "one"]


# 아래와 같이 할 수도 있다.

# In[63]:


df.loc["bar"].loc["one"]


# - 데이터프레임 인덱싱: 열 라벨이 다중 인덱스인 경우

# In[64]:


df1


# In[65]:


df1["bar"]


# 레벨별로 라벨을 지정한다. 각각의 라벨은 쉼표로 구분한다.

# In[66]:


df1["bar", "one"]


# 아래와 같이 할 수도 있다

# In[67]:


df1["bar"]["one"]


# ### 슬라이싱

# 다중 인덱스를 라벨로 사용하는 시리즈와 데이터프레임의 인덱싱은 일반 슬라이싱과 크게 다르지 않다.

# In[68]:


df


# - 0-레벨 인덱싱

# In[69]:


df.loc["baz":"foo"]


# - (0, 1)-레벨 인덱싱

# In[70]:


df.loc[("baz", "two"):("qux", "one")]


# 튜플들의 리스트를 지정하면 리인덱싱처럼 작동한다.

# In[71]:


df.loc[[("bar", "two"), ("qux", "one")]]


# 이외에 `slice()` 함수와 `pd.IndexSlice` 객체를 사용하는 방법도 있지만 여기서는 다루지 않는다.

# ## 그룹 분류: `pd.groupby()` 함수

# - 참고: [Grouping section](https://pandas.pydata.org/docs/user_guide/groupby.html#groupby)

# `pd.groupby()` 함수는 다음 3 기능을 제공한다.
# 
# - **분류**<font size='2'>Splitting</font>: 데이터를 조건에 따라 여러 그룹으로 분류
# - **함수 적용**<font size='2'>Applying</font>: 그룹별로 함수 적용
# - **조합**<font size='2'>Combining</font>: 그룹별 함수 적용 결과를 취합하여 새로운 데이터프레임/시리즈 생성

# In[72]:


df = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar',
                         'foo', 'bar', 'foo', 'bar'],
                   'B': ['one', 'one', 'two', 'three',
                         'two', 'two', 'one', 'three'],
                   'C': np.random.randn(8),
                   'D': np.random.randn(8)})

df


# - `A` 열에 사용된 항목 기준으로 그룹으로 분류한 후 그룹별로 `C`와 `D` 열의 모든 항목의 합 계산해서 새로운 데이터프레임 생성
#     <br><br>
#     
#     | `A` | 경우의 수 |
#     | :---: | :---: |
#     | `bar` | 1 |
#     | `foo` | 1 |

# In[73]:


df.groupby('A')[["C", "D"]].sum()


# - `A`열의 항목과 `B` 열의 항목의 조합을 기준으로 그룹으로 그룹별로 `C`와 `D` 열의 모든 항목의 합 계산해서 새로운 데이터프레임 생성
#     <br><br>
#     
#     | `A` | `B` | 경우의 수 |
#     | :---: | :---: | :---: |
#     | `bar` | `one`, `three`, `two` | 3 |
#     | `foo` | `one`, `two` | 2 |

# In[74]:


df.groupby(["A", "B"]).sum()


# **그룹 확인**

# - `for` 반복문 활용 

# In[75]:


for name, group in df.groupby(["A", "B"]):
    print(name)
    print(group)


# - `get_group()` 메서드

# In[76]:


df.groupby(["A", "B"]).get_group(('bar', 'one'))


# In[77]:


df.groupby(["A", "B"]).get_group(('bar', 'three'))


# - `groups` 속성

# In[78]:


df.groupby(["A", "B"]).groups


# - `value_counts` 속성

# In[79]:


df.groupby(["A", "B"]).value_counts()


# - `nunique` 속성

# In[80]:


df.groupby(["A", "B"]).nunique()


# - `sort=True` 키워드 인자

# In[81]:


df.groupby(["A", "B"], sort=True).sum()


# In[82]:


df.groupby(["A", "B"], sort=False).sum()


# In[83]:


df.groupby(["A", "B"], sort=False).nunique()


# **그룹 연산 예제**

# - `max()` 메서드

# In[84]:


df.groupby('A')[["C", "D"]].max()


# In[85]:


df.groupby(["A", "B"]).max()


# - `mean()` 메서드

# In[86]:


df.groupby('A')[["C", "D"]].mean()


# In[87]:


df.groupby(["A", "B"]).mean()


# - `size()` 메서드

# In[88]:


df.groupby('A')[["C", "D"]].size()


# In[89]:


df.groupby(["A", "B"]).size()


# - `describe()` 메서드

# In[90]:


df.groupby('A')[["C", "D"]].describe()


# In[91]:


df.groupby(["A", "B"]).describe()


# ## 모양 변환<font size='2'>Reshaping</font>

# - 참고: [Reshaping](https://pandas.pydata.org/docs/user_guide/reshaping.html#reshaping-stacking)

# ### 스택

# 열의 레벨을 하나 줄일 때 사용한다.
# 없어진 레벨은 열의 마지막 레벨로 추가된다.

# In[92]:


index


# In[93]:


df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=["A", "B"])
df


# In[94]:


df2 = df[:4]
df2


# - `stack()` 메서드: 
#     열이 한 개의 레벨로 구성되어 있기에 `stack()` 메서드를 적용하면
#     결국 모든 열이 없어지고, 열의 라벨은
#     인덱스의 마지막 레벨의 라벨로 변환된다.
#     여기서는 결국 3중 인덱스를 사용하는 시리즈를 생성한다.

# In[95]:


stacked = df2.stack()
stacked


# ### 언스택

# - `unstack()` 메서드: 
#     인덱스의 지정된 레벨을 열의 마지막 레벨로 변환한다.
#     인자를 지정하지 않으면 마지막 레벨을 변환한다.

# In[96]:


stacked.unstack()


# In[97]:


stacked.unstack().unstack()


# 인자를 지정하면 해당 레벨을 열의 마지막 레벨로 변환한다.

# In[98]:


stacked.unstack(0)


# In[99]:


stacked.unstack(1)


# ### 피버팅<font size='2'>Pivoting</font>

# - 참고: [Pivot Tables](https://pandas.pydata.org/docs/user_guide/reshaping.html#reshaping-pivot)

# **`pd.pivot_table()` 함수**

# In[100]:


import datetime

df = pd.DataFrame(
    {
        "A": ["one", "one", "two", "three"] * 6,
        "B": ["A", "B", "C"] * 8,
        "C": ["foo", "foo", "foo", "bar", "bar", "bar"] * 4,
        "D": np.random.randn(24),
        "E": np.random.randn(24),
        "F": [datetime.datetime(2013, i, 1) for i in range(1, 13)]
        + [datetime.datetime(2013, i, 15) for i in range(1, 13)],
    }
)

df


# In[101]:


pd.pivot_table(df, values="D", index=["A", "B"], columns=["C"]) # aggfunc=np.mean 이 기본값


# In[102]:


pd.pivot_table(df, values="D", index=["A", "B"], columns=["C"], aggfunc=np.sum)


# **`DataFrame.pivot()` 메서드**

# In[103]:


df


# In[104]:


df1 = df.groupby(['A', 'B']).sum().reset_index()
df1


# In[105]:


df1.pivot(columns='A', index='B', values="D")


# In[106]:


df1.pivot_table(columns='A', index='B', values="D")


# **예제**

# In[107]:


df = pd.DataFrame(
    {
        "A": ["one", "one", "two", "three"] * 3,
        "B": ["A", "B", "C"] * 4,
        "C": ["foo", "foo", "foo", "bar", "bar", "bar"] * 2,
        "D": np.random.randn(12),
        "E": np.random.randn(12),
    }
)
df


# [`pivot_table()`](https://pandas.pydata.org/docs/reference/api/pandas.pivot_table.html#pandas.pivot_table) pivots a [`DataFrame`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html#pandas.DataFrame) specifying the `values`, `index`, and `columns`
# 
# 

# In[108]:


pd.pivot_table(df, values="D", index=["A", "B"], columns=["C"])


# In[109]:


pd.pivot(df, values="D", index=["A", "B"], columns=["C"])


# In[110]:


df.pivot_table(values="D", index=["A", "B"], columns=["C"], aggfunc=np.mean)


# In[111]:


pd.pivot_table(df, values="D", index=["A", "B"], columns=["C"], aggfunc=np.sum)


# In[112]:


df.pivot_table(values="D", index=["A", "B"], columns=["C"], aggfunc=np.mean)


# In[113]:


pd.pivot_table(df, values="D", index=["A"], columns=["C"], aggfunc=np.mean)


# In[114]:


df.groupby(["A", "C"]).mean()


# In[115]:


df1 = df.groupby(["A", "C"]).mean().reset_index()
df1


# In[116]:


pd.pivot(df1, values="D", index=["A"], columns=["C"])


# In[ ]:





# In[ ]:





# In[ ]:





# ## Time series
# 
# pandas has simple, powerful, and efficient functionality for performing resampling operations during frequency conversion (e.g., converting secondly data into 5-minutely data). This is extremely common in, but not limited to, financial applications. See the [Time Series](https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries) section.

# In[117]:


rng = pd.date_range("1/1/2012", periods=100, freq="S")
rng


# In[118]:


ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)
ts


# In[119]:


ts.resample("10S").sum()


# In[120]:


ts.resample("1Min").sum()


# [`Series.tz_localize()`](https://pandas.pydata.org/docs/reference/api/pandas.Series.tz_localize.html#pandas.Series.tz_localize) localizes a time series to a time zone:

# In[121]:


rng = pd.date_range("3/6/2012 00:00", periods=5, freq="D")
ts = pd.Series(np.random.randn(len(rng)), rng)
print(ts, "\n")
ts_utc = ts.tz_localize("UTC")
ts_utc


# Converting between time span representations:

# In[122]:


rng = pd.date_range("1/1/2012", periods=5, freq="M")
rng


# In[123]:


ts = pd.Series(np.random.randn(len(rng)), index=rng)
ts


# In[124]:


ps = ts.to_period()
ps


# In[125]:


ps.to_timestamp()


# Converting between period and timestamp enables some convenient arithmetic functions to be used. In the following example, we convert a quarterly frequency with year ending in November to 9am of the end of the month following the quarter end:

# In[126]:


prng = pd.period_range("1990Q1", "2000Q4", freq="Q-NOV")
prng


# In[127]:


ts = pd.Series(np.random.randn(len(prng)), prng)
ts


# In[128]:


ts.index = (prng.asfreq("M", "e") + 1).asfreq("H", "s") + 9
ts.index


# In[129]:


ts.head()


# ## Categoricals
# 
# pandas can include categorical data in a [`DataFrame`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html#pandas.DataFrame). For full docs, see the [categorical introduction](https://pandas.pydata.org/docs/user_guide/categorical.html#categorical) and the [API documentation](https://pandas.pydata.org/docs/reference/arrays.html#api-arrays-categorical).

# In[130]:


df = pd.DataFrame(
    {"id": [1, 2, 3, 4, 5, 6], "raw_grade": ["a", "b", "b", "a", "a", "e"]}
)
df


# Converting the raw grades to a categorical data type:

# In[131]:


df["grade"] = df["raw_grade"].astype("category")
df["grade"]


# Rename the categories to more meaningful names:

# In[132]:


new_categories = ["very good", "good", "very bad"]
df["grade"] = df["grade"].cat.rename_categories(new_categories)
df


# Reorder the categories and simultaneously add the missing categories (methods under [`Series.cat()`](https://pandas.pydata.org/docs/reference/api/pandas.Series.cat.html#pandas.Series.cat) return a new [`Series`](https://pandas.pydata.org/docs/reference/api/pandas.Series.html#pandas.Series) by default):

# In[133]:


df["grade"] = df["grade"].cat.set_categories(
    ["very bad", "bad", "medium", "good", "very good"]
)
df["grade"]


# Sorting is per order in the categories, not lexical order:

# In[134]:


df.sort_values(by="grade")


# Grouping by a categorical column also shows empty categories:

# In[135]:


df.groupby("grade").size()


# ## Plotting
# 
# See the [Plotting](https://pandas.pydata.org/docs/user_guide/visualization.html#visualization) docs.
# 
# We use the standard convention for referencing the matplotlib API:

# In[136]:


import matplotlib.pyplot as plt
plt.close("all")

ts = pd.Series(np.random.randn(1000), index=pd.date_range("1/1/2000", periods=1000))
ts = ts.cumsum()
ts.plot();


# If running under Jupyter Notebook, the plot will appear on [`plot()`](https://pandas.pydata.org/docs/reference/api/pandas.Series.plot.html#pandas.Series.plot). Otherwise use [`matplotlib.pyplot.show`](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.show.html) to show it or [`matplotlib.pyplot.savefig`](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.savefig.html) to write it to a file.
# 
# On a DataFrame, the [`plot()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html#pandas.DataFrame.plot) method is a convenience to plot all of the columns with labels:

# In[137]:


df = pd.DataFrame(
    np.random.randn(1000, 4), index=ts.index, columns=["A", "B", "C", "D"]
)

df = df.cumsum()
plt.figure();
df.plot();
plt.legend(loc='best');


# ## Importing and exporting data

# ### CSV
# 
# [Writing to a csv file](https://pandas.pydata.org/docs/user_guide/io.html#io-store-in-csv): using [`DataFrame.to_csv()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html#pandas.DataFrame.to_csv)

# In[138]:


df.to_csv("foo.csv")


# [Reading from a csv file](https://pandas.pydata.org/docs/user_guide/io.html#io-read-csv-table): using [`read_csv()`](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html#pandas.read_csv)

# In[139]:


pd.read_csv("foo.csv")


# ### Excel
# 
# Reading and writing to [Excel](https://pandas.pydata.org/docs/user_guide/io.html#io-excel).
# 
# Writing to an excel file using [`DataFrame.to_excel()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_excel.html#pandas.DataFrame.to_excel):

# In[140]:


df.to_excel("foo.xlsx", sheet_name="Sheet1")


# Reading from an excel file using [`read_excel()`](https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html#pandas.read_excel):

# In[141]:


pd.read_excel("foo.xlsx", "Sheet1", index_col=None, na_values=["NA"])

