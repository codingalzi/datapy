#!/usr/bin/env python
# coding: utf-8

# (sec:pandas10min_2)=
# # 판다스 10분 완성 2부
# 

# **필수 라이브러리**

# In[1]:


import numpy as np
import pandas as pd


# ## 데이터 결합: merge-join-concat

# - 참고: [Merging section](https://pandas.pydata.org/docs/user_guide/merging.html#merging)

# ### 합종연횡: `pd.concat()` 함수

# `pd.concat()` 함수는 여러 개의 데이터프레임을 하나로 합친다.
# 
# - `axis=0`: 합종 결합

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


# - `axis=1`: 연횡 결합

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


# In[8]:


df1.index


# In[9]:


pd.concat([df1, df4], axis=1).reindex(df1.index)


# In[10]:


df4.reindex(df1.index)


# In[11]:


pd.concat([df1, df4.reindex(df1.index)], axis=1)


# ### 합병: `pd.merge()` 함수

# `pd.merge()` 함수 는 SQL 방식으로 특정 열을 기준으로 두 개의 데이터프레임을 합친다.
# 다양한 옵션을 지원하는 매우 강력한 도구이다.
# 
# - 참고: [Database style joining](https://pandas.pydata.org/docs/user_guide/merging.html#merging-join)

# **예제**

# 실습을 위해 아래 두 데이터프레임을 이용한다.

# In[12]:


left = pd.DataFrame({"key": ["foo", "foo"], "lval": [1, 2]})
right = pd.DataFrame({"key": ["foo", "foo"], "rval": [4, 5]})


# In[13]:


left


# In[14]:


right


# - `on="key"` 키워드 인자
#     - `key` 열에 사용된 항목 각각에 대해 다른 열에서 해당 항목과 연관된 값들을 조합할 수 있는 모든 경우의 수를 다룬다.
#     - `foo` 값에 대해 `lval` 열에서 2개의 값이,
#         `rval` 열에서 2개의 값이 있기에 `foo`와 관련해서 총 4개의 경우가 생성된다.
#         
#     | `key` | `left.lval` | `right.rval` | 경우의 수 |
#     | :---: | :---: | :---: | :---: |
#     | `foo` | `1, 2` | `4, 5` | 4 |

# In[15]:


pd.merge(left, right, on="key")


# **예제**

# In[16]:


left = pd.DataFrame({"key": ["foo", "bar"], "lval": [1, 2]})
right = pd.DataFrame({"key": ["foo", "bar"], "rval": [4, 5]})


# In[17]:


left


# In[18]:


right


# - `on="key"` 키워드 인자
#     - `key` 열에 사용된 항목별로 모든 경우의 수를 다룬다.
#     - `foo` 값에 대해 `lval` 열에서 1개의 값이,
#         `rval` 열에서 1개의 값이 있기에 `foo`와 관련해서 총 1개의 경우가 생성된다.
#     - `bar` 값에 대해 `lval` 열에서 1개의 값이,
#         `rval` 열에서 1개의 값이 있기에 `foo`와 관련해서 총 1개의 경우가 생성된다.
#         
#     | `key` | `left.lval` | `right.rval` | 경우의 수 |
#     | :---: | :---: | :---: | :---: |
#     | `foo` | `1` | `4` | 1 |        
#     | `bar` | `2` | `5` | 1 |        

# In[19]:


pd.merge(left, right, on="key")


# **예제**

# 경우의 수는 지정된 열의 항목이 사용된 횟수를 기준으로 한다. 

# In[20]:


left = pd.DataFrame(
    {
        "key": ["K0", "K1", "K2", "K3"],
        "A": ["A0", "A1", "A2", "A3"],
        "B": ["B0", "B1", "B2", "B3"],
    }
)


# In[21]:


left


# In[22]:


right = pd.DataFrame(
    {
        "key": ["K0", "K1", "K2", "K3"],
        "C": ["C0", "C1", "C2", "C3"],
        "D": ["D0", "D1", "D2", "D3"],
    }
)


# In[23]:


right


# | `key` | (`left.A`, `left.B`) | (`right.C`, `right.D`) | 경우의 수 |
# | :---: | :---: | :---: | :---: |
# | `K0` | (`A0`, `B0`) | (`C0`, `D0`) | 1 |
# | `K1` | (`A1`, `B1`) | (`C1`, `D1`) | 1 |
# | `K2` | (`A2`, `B2`) | (`C2`, `D2`) | 1 |
# | `K3` | (`A3`, `B3`) | (`C3`, `D3`) | 1 |

# In[24]:


result = pd.merge(left, right, on="key")
result


# **다양한 키워드 인자**

# In[25]:


left = pd.DataFrame(
    {
        "key1": ["K0", "K0", "K1", "K2"],
        "key2": ["K0", "K1", "K0", "K1"],
        "A": ["A0", "A1", "A2", "A3"],
        "B": ["B0", "B1", "B2", "B3"],
    }
)

left


# In[26]:


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

# In[27]:


result = pd.merge(left, right, on=["key1", "key2"]) # how='inner' 가 기본값
result


# In[28]:


result = pd.merge(left, right, how="inner", on=["key1", "key2"])
result


# - `how='outer'`: 지정된 키의 합집합 대상

# In[29]:


result = pd.merge(left, right, how="outer", on=["key1", "key2"])
result


# - `how='left'`: 왼쪽 데이터프레임의 키에 포함된 항목만 대상

# In[30]:


left


# In[31]:


result = pd.merge(left, right, how="left", on=["key1", "key2"])
result


# - `how='right'`: 오른쪽 데이터프레임의 키에 포함된 항목만 대상

# In[32]:


right


# In[33]:


result = pd.merge(left, right, how="right", on=["key1", "key2"])
result


# - `how='cross'`: 모든 경우의 수 조합

# In[34]:


result = pd.merge(left, right, how="cross")
result


# ### 합병: `DataFrame.join()` 메서드

# 인덱스를 기준으로 두 개의 데이터프레임을 합병할 때 사용한다.

# In[35]:


left = pd.DataFrame(
    {"A": ["A0", "A1", "A2"], "B": ["B0", "B1", "B2"]}, index=["K0", "K1", "K2"]
)

left


# In[36]:


right = pd.DataFrame(
    {"C": ["C0", "C2", "C3"], "D": ["D0", "D2", "D3"]}, index=["K0", "K2", "K3"]
)

right


# In[37]:


left.join(right)


# 아래와 같이 `pd.merge()` 함수를 이용한 결과와 동일하다.

# In[38]:


pd.merge(left, right, left_index=True, right_index=True, how='left')


# `pd.merge()` 함수의 키워드 인자를 동일하게 사용할 수 있다.

# - `how='outer'`

# In[39]:


left.join(right, how="outer")


# 아래 코드가 동일한 결과를 낸다.

# In[40]:


pd.merge(left, right, left_index=True, right_index=True, how='outer')


# - `how='inner'`

# In[41]:


left.join(right, how="inner")


# 아래 코드가 동일한 결과를 낸다.

# In[42]:


pd.merge(left, right, left_index=True, right_index=True, how='inner')


# ## 그룹화: `pd.groupby()` 함수

# - 참고: [Grouping section](https://pandas.pydata.org/docs/user_guide/groupby.html#groupby)

# `pd.groupby()` 함수는 다음 3 기능을 제공한다.
# 
# - **분류**<font size='2'>Splitting</font>: 데이터를 조건에 따라 여러 그룹으로 분류
# - **함수 적용**<font size='2'>Applying</font>: 그룹별로 함수 적용
# - **조합**<font size='2'>Combining</font>: 그룹별 함수 적용 결과를 취합하여 새로운 데이터프레임/시리즈 생성

# In[43]:


df = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar',
                         'foo', 'bar', 'foo', 'bar'],
                   'B': ['one', 'one', 'two', 'three',
                         'two', 'two', 'one', 'three'],
                   'C': np.random.randn(8),
                   'D': np.random.randn(8)})

df


# - `A` 열에 사용된 항목 기준으로 그룹으로 분류한 후 그룹별로 `C`와 `D` 열의 모든 항목의 합 계산해서 새로운 데이터프레임 생성

# | `A`(사용횟수) | 경우의 수 |
# | :---: | :---: |
# | `bar`(4) | 1 |
# | `foo`(4) | 1 |

# In[44]:


df.groupby('A')[["C", "D"]].sum()


# - `A`열의 항목과 `B` 열의 항목의 조합을 기준으로 그룹으로 그룹별로 `C`와 `D` 열의 모든 항목의 합 계산해서 새로운 데이터프레임 생성

# | `A`(사용횟수) | `B`(사용횟수) | 경우의 수 |
# | :---: | :---: | :---: |
# | `bar`(4) | `one`(1), `three`(2), `two`(1) | 3 |
# | `foo`(4) | `one`(2), `two`(2) | 2 |

# In[45]:


df.groupby(["A", "B"]).sum()


# **그룹 확인**

# - `for` 반복문 활용 

# In[46]:


for name, group in df.groupby(["A", "B"]):
    print(name)
    print(group)


# - `get_group()` 메서드

# In[47]:


df.groupby(["A", "B"]).get_group(('bar', 'one'))


# In[48]:


df.groupby(["A", "B"]).get_group(('bar', 'three'))


# - `groups` 속성

# In[49]:


df.groupby(["A", "B"]).groups


# - `value_counts` 속성

# In[50]:


df.groupby(["A", "B"]).value_counts()


# - `nunique` 속성

# In[51]:


df.groupby(["A", "B"]).nunique()


# - `sort=True` 키워드 인자

# In[52]:


df.groupby(["A", "B"], sort=True).sum()


# In[53]:


df.groupby(["A", "B"], sort=False).sum()


# In[54]:


df.groupby(["A", "B"], sort=False).nunique()


# **그룹 연산**

# In[55]:


df.groupby('A')[["C", "D"]].max()


# In[56]:


df.groupby(["A", "B"]).max()


# In[57]:


df.groupby('A')[["C", "D"]].mean()


# In[58]:


df.groupby(["A", "B"]).mean()


# In[59]:


df.groupby('A')[["C", "D"]].size()


# In[60]:


df.groupby(["A", "B"]).size()


# In[61]:


df.groupby('A')[["C", "D"]].describe()


# In[62]:


df.groupby(["A", "B"]).describe()


# ## Reshaping
# 
# See the sections on [Hierarchical Indexing](https://pandas.pydata.org/docs/user_guide/advanced.html#advanced-hierarchical) and [Reshaping](https://pandas.pydata.org/docs/user_guide/reshaping.html#reshaping-stacking).

# ### Stack

# In[63]:


tuples = list(
    zip(
        ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
        ["one", "two", "one", "two", "one", "two", "one", "two"],
    )
)

tuples


# In[64]:


index = pd.MultiIndex.from_tuples(tuples, names=["first", "second"])
index


# In[65]:


df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=["A", "B"])
df


# In[66]:


df2 = df[:4]
df2


# The [`stack()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.stack.html#pandas.DataFrame.stack) method “compresses” a level in the DataFrame’s columns:
# 
# 

# In[67]:


stacked = df2.stack()
stacked


# With a “stacked” DataFrame or Series (having a [MultiIndex](https://pandas.pydata.org/docs/reference/api/pandas.MultiIndex.html#pandas.MultiIndex) as the `index`), the inverse operation of [`stack()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.stack.html#pandas.DataFrame.stack) is [`unstack()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.unstack.html#pandas.DataFrame.unstack), which by default unstacks the **last level**:

# In[68]:


stacked.unstack()


# In[69]:


stacked.unstack(1)


# In[70]:


stacked.unstack(0)


# ### Pivot tables
# 
# See the section on [Pivot Tables](https://pandas.pydata.org/docs/user_guide/reshaping.html#reshaping-pivot).

# In[71]:


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

# In[72]:


pd.pivot_table(df, values="D", index=["A", "B"], columns=["C"])


# ## Time series
# 
# pandas has simple, powerful, and efficient functionality for performing resampling operations during frequency conversion (e.g., converting secondly data into 5-minutely data). This is extremely common in, but not limited to, financial applications. See the [Time Series](https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries) section.

# In[73]:


rng = pd.date_range("1/1/2012", periods=100, freq="S")
rng


# In[74]:


ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)
ts


# In[75]:


ts.resample("10S").sum()


# In[76]:


ts.resample("1Min").sum()


# [`Series.tz_localize()`](https://pandas.pydata.org/docs/reference/api/pandas.Series.tz_localize.html#pandas.Series.tz_localize) localizes a time series to a time zone:

# In[77]:


rng = pd.date_range("3/6/2012 00:00", periods=5, freq="D")
ts = pd.Series(np.random.randn(len(rng)), rng)
print(ts, "\n")
ts_utc = ts.tz_localize("UTC")
ts_utc


# Converting between time span representations:

# In[78]:


rng = pd.date_range("1/1/2012", periods=5, freq="M")
rng


# In[79]:


ts = pd.Series(np.random.randn(len(rng)), index=rng)
ts


# In[80]:


ps = ts.to_period()
ps


# In[81]:


ps.to_timestamp()


# Converting between period and timestamp enables some convenient arithmetic functions to be used. In the following example, we convert a quarterly frequency with year ending in November to 9am of the end of the month following the quarter end:

# In[82]:


prng = pd.period_range("1990Q1", "2000Q4", freq="Q-NOV")
prng


# In[83]:


ts = pd.Series(np.random.randn(len(prng)), prng)
ts


# In[84]:


ts.index = (prng.asfreq("M", "e") + 1).asfreq("H", "s") + 9
ts.index


# In[85]:


ts.head()


# ## Categoricals
# 
# pandas can include categorical data in a [`DataFrame`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html#pandas.DataFrame). For full docs, see the [categorical introduction](https://pandas.pydata.org/docs/user_guide/categorical.html#categorical) and the [API documentation](https://pandas.pydata.org/docs/reference/arrays.html#api-arrays-categorical).

# In[86]:


df = pd.DataFrame(
    {"id": [1, 2, 3, 4, 5, 6], "raw_grade": ["a", "b", "b", "a", "a", "e"]}
)
df


# Converting the raw grades to a categorical data type:

# In[87]:


df["grade"] = df["raw_grade"].astype("category")
df["grade"]


# Rename the categories to more meaningful names:

# In[88]:


new_categories = ["very good", "good", "very bad"]
df["grade"] = df["grade"].cat.rename_categories(new_categories)
df


# Reorder the categories and simultaneously add the missing categories (methods under [`Series.cat()`](https://pandas.pydata.org/docs/reference/api/pandas.Series.cat.html#pandas.Series.cat) return a new [`Series`](https://pandas.pydata.org/docs/reference/api/pandas.Series.html#pandas.Series) by default):

# In[89]:


df["grade"] = df["grade"].cat.set_categories(
    ["very bad", "bad", "medium", "good", "very good"]
)
df["grade"]


# Sorting is per order in the categories, not lexical order:

# In[90]:


df.sort_values(by="grade")


# Grouping by a categorical column also shows empty categories:

# In[91]:


df.groupby("grade").size()


# ## Plotting
# 
# See the [Plotting](https://pandas.pydata.org/docs/user_guide/visualization.html#visualization) docs.
# 
# We use the standard convention for referencing the matplotlib API:

# In[92]:


import matplotlib.pyplot as plt
plt.close("all")

ts = pd.Series(np.random.randn(1000), index=pd.date_range("1/1/2000", periods=1000))
ts = ts.cumsum()
ts.plot();


# If running under Jupyter Notebook, the plot will appear on [`plot()`](https://pandas.pydata.org/docs/reference/api/pandas.Series.plot.html#pandas.Series.plot). Otherwise use [`matplotlib.pyplot.show`](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.show.html) to show it or [`matplotlib.pyplot.savefig`](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.savefig.html) to write it to a file.
# 
# On a DataFrame, the [`plot()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html#pandas.DataFrame.plot) method is a convenience to plot all of the columns with labels:

# In[93]:


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

# In[94]:


df.to_csv("foo.csv")


# [Reading from a csv file](https://pandas.pydata.org/docs/user_guide/io.html#io-read-csv-table): using [`read_csv()`](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html#pandas.read_csv)

# In[95]:


pd.read_csv("foo.csv")


# ### Excel
# 
# Reading and writing to [Excel](https://pandas.pydata.org/docs/user_guide/io.html#io-excel).
# 
# Writing to an excel file using [`DataFrame.to_excel()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_excel.html#pandas.DataFrame.to_excel):

# In[96]:


df.to_excel("foo.xlsx", sheet_name="Sheet1")


# Reading from an excel file using [`read_excel()`](https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html#pandas.read_excel):

# In[97]:


pd.read_excel("foo.xlsx", "Sheet1", index_col=None, na_values=["NA"])

