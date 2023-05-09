#!/usr/bin/env python
# coding: utf-8

# # 판다스 10분 완성
# 

# 판다스 초보자들을 위한 판다스 기초 내용을 다룬다.
# 여기서 다루는 배용은 [판다스 요리책](https://pandas.pydata.org/docs/user_guide/cookbook.html#cookbook)을 
# 많이 참고한다.

# **필수 라이브러리**

# In[1]:


import numpy as np
import pandas as pd


# ## 객체 생성

# 참고: [판다스 자료구조](https://pandas.pydata.org/docs/user_guide/dsintro.html#dsintro)

# **시리즈 객체 생성**

# 리스트를 이용하여 시리즈를 생성할 수 있다.

# In[2]:


s = pd.Series([1, 3, 5, np.nan, 6, 8])
s


# **데이터프레임 객체 생성**

# 방식 1: 2차원 어레이, 인덱스 라벨, 열 라벨을 지정하여 데이터프레임을 생성할 수 있다.

# - 인덱스 라벨: 날짜시간(`datetime`) 인덱스 이용

# In[3]:


dates = pd.date_range(start="20130101", periods=6)
dates


# - 열 라벨은 A, B, C, D로 지정

# In[4]:


df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list("ABCD"))
df


# 방식 2: 사전 객체를 이용할 수도 있다.
# - 사전의 키; 열 라벨
# - 인덱스 라벨: 정수 인덱스 자동 지정

# In[5]:


df2 = pd.DataFrame(
    {
        "A": 1.0,
        "B": pd.Timestamp("20130102"),
        "C": pd.Series(1, index=list(range(4)), dtype="float32"),
        "D": np.array([3] * 4, dtype="int32"),
        "E": pd.Categorical(["test", "train", "test", "train"]),
        "F": "foo",
    }
)

df2


# 열별로 다른 자료형이 사용될 수 있다.

# In[6]:


df2.dtypes


# ## 데이터 살펴보기

# 참고: [데이터프레임 핵심 기초](https://pandas.pydata.org/docs/user_guide/basics.html#basics)

# - 처음 5행 확인

# In[7]:


df.head()


# - 끝에서 3행 확인

# In[8]:


df.tail(3)


# - 인덱스 라벨 확인

# In[9]:


df.index


# - 열 라벨 확인

# In[10]:


df.columns


# - 넘파이 어레이로 변환: 인덱스 라벨과 열 라벨 정보 삭제

# In[11]:


df.to_numpy()


# - 열별 자료형이 통일되지 않은 경우: `object`로 통일된 자료형 사용. 시간 소요.

# In[12]:


df2.to_numpy()


# - 수치형 데이터의 분포 확인

# In[13]:


df.describe()


# - 전치 데이터프레임

# In[14]:


df.T


# - 열 라벨 내림차순 정렬

# In[15]:


df.sort_index(axis=1, ascending=False)


# - 특정 열의 값을 기준으로 행 정렬

# In[16]:


df.sort_values(by='B')


# ## 인덱싱/슬라이싱

# In[17]:


type(pd.core.indexing._LocIndexer)


# In[18]:


type(abs)


# In[19]:


np.r_


# In[20]:


type(np.c_)


# 권장 사항: 넘파이 어레이의 인덱싱, 슬라이싱 방식보다 아래 방식 권장됨. 보다 효율적이고 빠름.
# 
# - [`DataFrame.at[]`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.at.html#pandas.DataFrame.at)
# - [`DataFrame.iat[]`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iat.html#pandas.DataFrame.iat)
# - [`DataFrame.loc[]`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.loc.html#pandas.DataFrame.loc)
# - [`DataFrame.iloc[]`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iloc.html#pandas.DataFrame.iloc)
# 
# 참고
# 
# - [Indexing and Selecting Data](https://pandas.pydata.org/docs/user_guide/indexing.html#indexing)
# - [MultiIndex / Advanced Indexing](https://pandas.pydata.org/docs/user_guide/advanced.html#advanced)

# ### 열 선택

# 열 라벨을 이용한 인덱싱. 시리즈 생성.

# In[21]:


df["A"]


# 객체의 속성처럼 이용하는 방식도 가능.

# In[22]:


df.A


# ### 행 슬라이싱

# 정수 인덱스 활용. 데이터프레임 생성

# In[23]:


df[0:3]


# 인덱스 라벨 활용. 정수 인덱스 방식과 조금 다름.

# In[24]:


df["20130101":"20130103"]


# ### `loc[]`: 라벨 활용 인덱싱/슬라이싱

# 인덱스 라벨을 이용하면 열 라벨을 인덱스로 사용하는 시리즈가 생성된다.

# In[25]:


dates[0]


# In[26]:


df.loc[dates[0]]


# 축 활용. 행과 열에 대한 인덱싱/슬라이싱 동시에 지정.

# - `A`, `B` 두 열만 추출.

# In[27]:


df.loc[:, ["A", "B"]]


# - 특정 행만 대상으로 `A`, `B` 두 열 추출

# In[28]:


df.loc["20130102":"20130104", ["A", "B"]]


# 인덱싱이 사용되면 차원이 줄어듦.

# In[29]:


df.loc["20130102", ["A", "B"]]


# 두 개의 인덱싱은 결국 하나의 상수(스칼라) 생성.

# In[30]:


df.loc[dates[0], "A"]


# 하나의 항목을 선택할 때 `at` 함수 사용.

# In[31]:


df.at[dates[0], "A"]


# ### `iloc[]`: 정수 인덱스 활용 인뎅식/슬라이싱

# 행 선택

# In[32]:


df.iloc[3]


# 어레이 인덱싱/슬라이싱 방식

# In[33]:


df.iloc[3:5, 0:2]


# 넘파이 어레이의 팬시 인덱싱과는 다르게 작동한다.

# In[34]:


df.iloc[[1, 2, 4], [0, 2]]


# In[35]:


df.iloc[[1, 2, 4], [0, 2, 3]]


# In[36]:


df.iloc[[1, 2, 4], [0, 2, 3, 1]]


# 행 슬라이싱

# In[37]:


df.iloc[1::2, :]


# 열 슬라이싱

# In[38]:


df.iloc[:, 1:3]


# 하나의 항목 추출

# In[39]:


df.iloc[1, 1]


# `iat[]` 활용도 가능

# In[40]:


df.iat[1, 1]


# ### 부울 인덱싱

# 마스크 활용

# - `A` 열에 양수 항목이 있는 행만 추출

# In[41]:


df["A"] > 0


# In[42]:


df[df["A"] > 0]


# 양수 항목만 그대로 두고 나머지는 결측치로 처리

# In[43]:


df[df > 0]


# 넘파이 어레이 방식과 다르게 작동한다.
# 아래 코드에서처럼 양수 항목만 모은 1차원 어레이가 생성된다.

# In[44]:


aArray = df.to_numpy()
aArray


# In[45]:


aArray[aArray > 0]


# - [`isin()` 메서드](https://pandas.pydata.org/docs/reference/api/pandas.Series.isin.html#pandas.Series.isin) 활용

# In[46]:


df2 = df.copy()
df2["E"] = ["one", "one", "two", "three", "four", "three"]
df2


# `E` 열에 `"two"` 또는 `"four"` 가 항목으로 사용된 행만 `True`

# In[47]:


df2["E"].isin(["two", "four"])


# `E` 열에 `"two"` 또는 `"four"` 가 항목으로 사용된 행만 추출하기

# In[48]:


df2[df2["E"].isin(["two", "four"])]


# ### 항목 지정

# In[49]:


s1 = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range("20130102", periods=6))
s1


# `F` 열 추가. 항목은 `s1` 이용.
# 0번 행은 결측치로 처리됨.

# In[50]:


df["F"] = s1
df


# `at[]` 활용: 첫재 행, `A` 열 항목을 0으로 지정.

# In[51]:


df.at[dates[0], "A"] = 0

df


# `iat[]`도 활용 가능.

# In[52]:


df.iat[0, 1] = 0


# 어레이를 이용하여 열 또는 행을 지정할 수 있다.

# - `D` 열 항목 지정

# In[53]:


df.loc[:, "D"] = np.array([5] * len(df))


# In[54]:


df


# - 1번 행 항목 지정

# In[55]:


df.loc[dates[1], :] = np.array([3] * df.shape[1])


# In[56]:


df


# - `iloc[]` 도 사용 가능

# In[57]:


df.iloc[2, :] = np.array([4] * df.shape[1])


# In[58]:


df


# ### `where()`/`mask()` 메서드 활용

# 참고
# 
# - [`DataFrame.where()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.where.html)
# - [`DataFrame.mask()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.mask.html)

# In[59]:


s = pd.Series(range(5))


# `where(조건식)`은 시리즈/데이터프레임의 항목 중에서 조건식이
# 거짓이 되도록 하는 항목 모두 결측치로 처리한 시리즈/데이터프레임을 생성한다.

# In[60]:


s.where(s > 0)


# In[61]:


df.where(df > 0)


# `mask(조건식)`은 시리즈/데이터프레임의 항목 중에서 조건식이
# 참이 되도록 하는 항목 모두 결측치로 처리한 시리즈/데이터프레임을 생성한다.

# In[62]:


s.mask(s > 0)


# In[63]:


s.mask(s <= 0)


# In[64]:


df.mask(df > 0)


# In[65]:


df.mask(df <= 0)


# `where()`/`mask()`의 인자로 부울 시리즈 또는 부울 데이터프레임이 사용될 수 있다.
# 그러면 `True`가 위치한 곳만 대상으로 마스크가 작동한다.

# In[66]:


t = pd.Series([True, False, False, True])

t


# 0번, 3번 위치만 참으로 처리된다.
# 4번 위치처럼 마스크에서 아예 위치로 언급되지 않는 경우는 무조건 거짓으로 처리된다.

# In[67]:


s.where(t)


# `mask()` 메서드는 4번 위치처럼 마스크에서 언급되지 않은 곳은 무조건 참으로 처리한다.

# In[68]:


s.mask(t)


# In[69]:


df2 = df.copy()
df2


# 연습을 위해 결측치를 제거한다.

# In[70]:


df2.iloc[0, -1] = 4.
df2


# `where()`/`mask()`가 두 개의 인자를 사용하면
# 조건식이 참/거짓이 되는 항목을 결측치가 아닌 둘째 인자로 대체한다.

# - 양수 항목은 해당 값의 음수로 대체

# In[71]:


df3 = df2.where(df2 > 0, -df2)
df3


# 아래처럼 부울 인덱싱하는 것과 동일하다.

# In[72]:


df4 = df2.copy()
df4[df2 <= 0] = -df2
df4


# In[73]:


(df3 == df4).all(axis=None)


# `mask()` 메서드도 유사하게 작동한다.

# In[74]:


df5 = df2.mask(df2 <= 0, -df2)


# In[75]:


(df3 == df5).all(None)  # axis=None


# ## 결측치

# - 내부적으로 `np.nan`을 사용. 겉으로는 자료형에 따라 
#     `NaN`(부동소수점), `NA`(정수), `NaT`(시간) 등으로 표기.
# - 참고: [Missing Data](https://pandas.pydata.org/docs/user_guide/missing_data.html#missing-data)

# 결측치가 포함된 어떤 연산도 결측치로 처리된다.

# In[76]:


np.nan + 1


# 심지어 두 결측치의 비교도  허영 안된다.

# In[77]:


np.nan == np.nan


# 반면에 `None`은 하나의 값으로 간주되어 비교가 가능하다.

# In[78]:


None == None


# 만약 적절하게 사용하지 않으면 오류가 발생한다.

# ```python
# >>> None + 1
# ---------------------------------------------------------------------------
# TypeError                                 Traceback (most recent call last)
# Cell In [101], line 1
# ----> 1 None + 1
# 
# TypeError: unsupported operand type(s) for +: 'NoneType' and 'int'
# ```

# ### 결측치 처리

# 연습을 위해 결측치를 일부 포함한 데이터프레임을 생성한다.

# In[79]:


df


# `reindex()` 메서드를 이용하여 행과 열의 라벨을 새로 지정한다.

# In[80]:


df1 = df.reindex(index = dates[0:4], columns = list(df.columns) + ['E'])
df1


# 결측치 일부를 채운다.

# In[81]:


df1.loc[dates[0] : dates[1], 'E'] = 1
df1


# - [`DataFrame.dropna()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html#pandas.DataFrame.dropna) 메서드: 결측치를 포함한 행을 삭제한 데이터프레임 생성
# 
# 

# In[82]:


df1.dropna(how='any')


# - [`DataFrame.fillna()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.fillna.html#pandas.DataFrame.fillna): 결측치를 지정된 값으로 채운 데이터프레임 생성

# In[83]:


df1.fillna(value=5)


# - [`isna()`](https://pandas.pydata.org/docs/reference/api/pandas.isna.html#pandas.isna):
#     결측치가 위치한 곳만 `True`로 처리하는 부울 마스크 생성

# In[84]:


pd.isna(df)


# ## 연산

# - 참고: [Basic section on Binary Ops](https://pandas.pydata.org/docs/user_guide/basics.html#basics-binop).

# ### 통계

# 주의사항
# 
# - 결측치는 무시된다.

# In[85]:


df


# `F` 열은 결측치를 제외한 5개의 값의 평균값을 구한다.

# In[86]:


df.mean()


# 실제로 `F` 열에서 결측치를 제외한 항목 개수는 5이다.

# In[87]:


df.F.value_counts()


# In[88]:


df.F.value_counts().sum()


# 결측치를 제외한 항목의 합을 5로 나눈 값은 3.8이다.

# In[89]:


df.F.sum()/5


# **축 활용**
# 
# 축을 지정하면 행 또는 열 기준으로 작동한다.

# - 행별 평균값

# In[90]:


df.mean(1) # axis=1


# ### 사칙연산

# In[91]:


s = pd.Series([1, 3, 5, np.nan, 6, 8], index = dates)
s


# 결측치를 더 추가한다.
# 
# 참고; [`DataFrame.shift()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.shift.html)

# In[92]:


s = pd.Series([1, 3, 5, np.nan, 6, 8], index = dates).shift(2)
s


# - 행별 뺄셈: `df - s`

# In[93]:


df


# 결츠치가 관여하면 무조건 결측치로 처리된다.

# In[94]:


df.sub(s, axis='index') # axis=0


# 브로드캐스팅은 필요에 따라 자동 적용된다.

# In[95]:


df - 5


# ### 함수 적용

# - 참고: [`DataFrame.apply()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.apply.html#pandas.DataFrame.apply)

# In[96]:


df


# - 열별 누적합

# In[97]:


df.apply(np.cumsum)


# - 행별 누적합

# In[98]:


df.apply(np.cumsum, axis='columns') # axis=1


# - 열별 최대값과 최소값의 차이. 결측치는 무시

# In[99]:


df.apply(lambda x: x.max() - x.min())


# - 행별 최대값과 최소값의 차이. 결측치는 무시

# In[100]:


df.apply(lambda x: x.max() - x.min(), axis=1)


# ### 이산화

# 참고: [Histogramming and Discretization](https://pandas.pydata.org/docs/user_guide/basics.html#basics-discretization)

# In[101]:


np.random.seed(17)

arr = pd.Series(np.random.randn(20))
arr


# `hist()` 메서드는 값의 범위를 10등분해서 각 구간에 속한 값들의 개수를 히스토그램으로 보여준다.

# In[102]:


arr.hist() # bins=10 이 기본


# 전체 값의 범위를 4등분한 다음에 막대그래프를 그려보자.

# In[103]:


factor = pd.cut(arr, bins=4)
factor


# 구간별 항목의 개수 확인

# In[104]:


factor.value_counts()


# 막대그래프 그리기

# In[105]:


factor.value_counts().sort_index().plot.bar(rot=0, grid=True)


# 4등분한 구간에 라벨을 붙이면 정보를 보다 정확히 전달한다.

# In[106]:


factor = pd.cut(arr, bins=4, labels=['A', 'B', 'C', 'D'])
factor


# In[107]:


factor.value_counts().sort_index().plot.bar(rot=0, grid=True)


# ### 문자열 메서드 활용

# 참고
# 
# - [정규식](https://docs.python.org/3/library/re.html)
# - [벡터와 문자열 메서드](https://pandas.pydata.org/docs/user_guide/text.html#text-string-methods)

# `str` 속성은 모든 항목을 문자열로 변환한 벡터를 가리킨다.

# In[108]:


s = pd.Series(["A", "B", "C", "Aaba", "Baca", np.nan, "CABA", "dog", "cat"])
s.str


# 변환된 벡터에 문자열 메서드를 적용하면 새로운 시리즈가 생성된다.

# In[109]:


s.str.lower()


# In[110]:


df = pd.DataFrame(np.random.randn(3, 2), columns=[" Column A ", " Column B "], index=range(3))
df


# 열 라벨에 대해 문자열 메서드를 적용해보자.

# - 소문자화

# In[111]:


df.columns.str.lower()


# - 양끝의 공백 제거

# In[112]:


df.columns.str.lower().str.strip()


# - 중간에 위치한 공백을 밑줄(underscore)로 대체

# In[113]:


df.columns.str.strip().str.lower().str.replace(" ", "_")


# ## 데이터 결합: merge-join-concat

# - 참고: [Merging section](https://pandas.pydata.org/docs/user_guide/merging.html#merging)

# ### 이어붙이기: `pd.concat()` 함수

# `pd.concat()` 함수는 여러 개의 데이터프레임을 하나로 합친다.
# 
# 아래 코드는 실습을 위해 임의로 생성된 데이터프레임을 세 개로 쪼갠다.

# In[114]:


df = pd.DataFrame(np.random.randn(10, 4))
df


# In[115]:


pieces = [df[:3], df[3:7], df[7:]]


# 아래 코드는 쪼갠 3 개의 데이터프레임을 **횡으로 합쳐**, 즉 열을 추가하는 방식으로
# 원래의 데이터프레임과 동일한 데이터프레임을 생성한다.

# In[116]:


pd.concat(pieces)


# ### 합병: `pd.merge()` 함수

# `pd.join()` 함수 는 SQL 방식으로 특정 열을 기준으로 두 개의 데이터프레임을 합친다.
# 다양한 옵션을 지원하는 매우 강력한 도구이다.
# 
# - 참고: [Database style joining](https://pandas.pydata.org/docs/user_guide/merging.html#merging-join)

# **예제**

# 실습을 위해 아래 두 데이터프레임을 이용한다.

# In[117]:


left = pd.DataFrame({"key": ["foo", "foo"], "lval": [1, 2]})
right = pd.DataFrame({"key": ["foo", "foo"], "rval": [4, 5]})


# In[118]:


left


# In[119]:


right


# - `on="key"` 키워드 인자
#     - `key` 열에 사용된 항목 각각에 대해 다른 열에서 해당 항목과 연관된 값들을 조합할 수 있는 모든 경우의 수를 다룬다.
#     - `foo` 값에 대해 `lval` 열에서 2개의 값이,
#         `rval` 열에서 2개의 값이 있기에 `foo`와 관련해서 총 4개의 경우가 생성된다.
#         
#     | `key` | `left.lval` | `right.rval` | 경우의 수 |
#     | :---: | :---: | :---: | :---: |
#     | `foo` | `1, 2` | `4, 5` | 4 |

# In[120]:


pd.merge(left, right, on="key")


# **예제**

# In[121]:


left = pd.DataFrame({"key": ["foo", "bar"], "lval": [1, 2]})
right = pd.DataFrame({"key": ["foo", "bar"], "rval": [4, 5]})


# In[122]:


left


# In[123]:


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

# In[124]:


pd.merge(left, right, on="key")


# **예제**

# 경우의 수는 지정된 열의 항목이 사용된 횟수를 기준으로 한다. 

# In[125]:


left = pd.DataFrame(
    {
        "key": ["K0", "K1", "K2", "K3"],
        "A": ["A0", "A1", "A2", "A3"],
        "B": ["B0", "B1", "B2", "B3"],
    }
)


# In[126]:


left


# In[127]:


right = pd.DataFrame(
    {
        "key": ["K0", "K1", "K2", "K3"],
        "C": ["C0", "C1", "C2", "C3"],
        "D": ["D0", "D1", "D2", "D3"],
    }
)


# In[128]:


right


# | `key` | (`left.A`, `left.B`) | (`right.C`, `right.D`) | 경우의 수 |
# | :---: | :---: | :---: | :---: |
# | `K0` | (`A0`, `B0`) | (`C0`, `D0`) | 1 |
# | `K1` | (`A1`, `B1`) | (`C1`, `D1`) | 1 |
# | `K2` | (`A2`, `B2`) | (`C2`, `D2`) | 1 |
# | `K3` | (`A3`, `B3`) | (`C3`, `D3`) | 1 |

# In[129]:


result = pd.merge(left, right, on="key")
result


# **다양한 키워드 인자**

# In[130]:


left = pd.DataFrame(
    {
        "key1": ["K0", "K0", "K1", "K2"],
        "key2": ["K0", "K1", "K0", "K1"],
        "A": ["A0", "A1", "A2", "A3"],
        "B": ["B0", "B1", "B2", "B3"],
    }
)

left


# In[131]:


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

# In[132]:


result = pd.merge(left, right, on=["key1", "key2"]) # how='inner' 가 기본값
result


# In[133]:


result = pd.merge(left, right, how="inner", on=["key1", "key2"])
result


# - `how='outer'`: 지정된 키의 합집합 대상

# In[134]:


result = pd.merge(left, right, how="outer", on=["key1", "key2"])
result


# - `how='left'`: 왼쪽 데이터프레임의 키에 포함된 항목만 대상

# In[135]:


left


# In[136]:


result = pd.merge(left, right, how="left", on=["key1", "key2"])
result


# - `how='right'`: 오른쪽 데이터프레임의 키에 포함된 항목만 대상

# In[137]:


right


# In[138]:


result = pd.merge(left, right, how="right", on=["key1", "key2"])
result


# - `how='cross'`: 모든 경우의 수 조합

# In[139]:


result = pd.merge(left, right, how="cross")
result


# ## 그룹화: `pd.groupby()` 함수

# - 참고: [Grouping section](https://pandas.pydata.org/docs/user_guide/groupby.html#groupby)

# `pd.groupby()` 함수는 다음 3 기능을 제공한다.
# 
# - **분류**: 데이터를 조건에 따라 여러 그룹으로 분류
# - **함수 적용**: 그룹별로 함수 적용
# - **조합**: 그룹별 함수 결과를 조합하여 새로운 데이터프레임/시리즈 생성

# In[140]:


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

# In[141]:


df.groupby('A')[["C", "D"]].sum()


# - `A`열의 항목과 `B` 열의 항목의 조합을 기준으로 그룹으로 그룹별로 `C`와 `D` 열의 모든 항목의 합 계산해서 새로운 데이터프레임 생성

# | `A`(사용횟수) | `B`(사용횟수) | 경우의 수 |
# | :---: | :---: | :---: |
# | `bar`(4) | `one`(2), `two`(2) | 2 |
# | `foo`(4) | `one`(1), `three`(2), `two`(1) | 3 |

# In[142]:


df.groupby(["A", "B"]).sum()


# **그룹 확인**

# - `for` 반복문 활용 

# In[143]:


for name, group in df.groupby(["A", "B"]):
    print(name)
    print(group)


# - `get_group()` 메서드

# In[144]:


df.groupby(["A", "B"]).get_group(('bar', 'one'))


# In[145]:


df.groupby(["A", "B"]).get_group(('bar', 'three'))


# - `groups` 속성

# In[146]:


df.groupby(["A", "B"]).groups


# - `value_counts` 속성

# In[147]:


df.groupby(["A", "B"]).value_counts()


# - `nunique` 속성

# In[148]:


df.groupby(["A", "B"]).nunique()


# - `sort=True` 키워드 인자

# In[149]:


df.groupby(["A", "B"], sort=True).sum()


# In[150]:


df.groupby(["A", "B"], sort=False).sum()


# In[151]:


df.groupby(["A", "B"], sort=False).nunique()


# **그룹 연산**

# In[152]:


df.groupby('A')[["C", "D"]].max()


# In[153]:


df.groupby(["A", "B"]).max()


# In[154]:


df.groupby('A')[["C", "D"]].mean()


# In[155]:


df.groupby(["A", "B"]).mean()


# In[156]:


df.groupby('A')[["C", "D"]].size()


# In[157]:


df.groupby(["A", "B"]).size()


# In[158]:


df.groupby('A')[["C", "D"]].describe()


# In[159]:


df.groupby(["A", "B"]).describe()


# In[ ]:





# ## Reshaping
# 
# See the sections on [Hierarchical Indexing](https://pandas.pydata.org/docs/user_guide/advanced.html#advanced-hierarchical) and [Reshaping](https://pandas.pydata.org/docs/user_guide/reshaping.html#reshaping-stacking).

# ### Stack

# In[160]:


tuples = list(
    zip(
        ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
        ["one", "two", "one", "two", "one", "two", "one", "two"],
    )
)

index = pd.MultiIndex.from_tuples(tuples, names=["first", "second"])

df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=["A", "B"])
df2 = df[:4]
df2


# The [`stack()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.stack.html#pandas.DataFrame.stack) method “compresses” a level in the DataFrame’s columns:
# 
# 

# In[161]:


stacked = df2.stack()
stacked


# With a “stacked” DataFrame or Series (having a [MultiIndex](https://pandas.pydata.org/docs/reference/api/pandas.MultiIndex.html#pandas.MultiIndex) as the `index`), the inverse operation of [`stack()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.stack.html#pandas.DataFrame.stack) is [`unstack()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.unstack.html#pandas.DataFrame.unstack), which by default unstacks the **last level**:

# In[162]:


stacked.unstack()


# In[163]:


stacked.unstack(1)


# In[164]:


stacked.unstack(0)


# ### Pivot tables
# 
# See the section on [Pivot Tables](https://pandas.pydata.org/docs/user_guide/reshaping.html#reshaping-pivot).

# In[165]:


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

# In[166]:


pd.pivot_table(df, values="D", index=["A", "B"], columns=["C"])


# ## Time series
# 
# pandas has simple, powerful, and efficient functionality for performing resampling operations during frequency conversion (e.g., converting secondly data into 5-minutely data). This is extremely common in, but not limited to, financial applications. See the [Time Series](https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries) section.

# In[167]:


rng = pd.date_range("1/1/2012", periods=100, freq="S")
ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)
ts.resample("5Min").sum()


# [`Series.tz_localize()`](https://pandas.pydata.org/docs/reference/api/pandas.Series.tz_localize.html#pandas.Series.tz_localize) localizes a time series to a time zone:

# In[168]:


rng = pd.date_range("3/6/2012 00:00", periods=5, freq="D")
ts = pd.Series(np.random.randn(len(rng)), rng)
print(ts, "\n")
ts_utc = ts.tz_localize("UTC")
ts_utc


# Converting between time span representations:

# In[169]:


rng = pd.date_range("1/1/2012", periods=5, freq="M")
ts = pd.Series(np.random.randn(len(rng)), index=rng)
ps = ts.to_period()
ps


# In[170]:


ps.to_timestamp()


# Converting between period and timestamp enables some convenient arithmetic functions to be used. In the following example, we convert a quarterly frequency with year ending in November to 9am of the end of the month following the quarter end:

# In[171]:


prng = pd.period_range("1990Q1", "2000Q4", freq="Q-NOV")
ts = pd.Series(np.random.randn(len(prng)), prng)
ts.index = (prng.asfreq("M", "e") + 1).asfreq("H", "s") + 9
ts.head()


# ## Categoricals
# 
# pandas can include categorical data in a [`DataFrame`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html#pandas.DataFrame). For full docs, see the [categorical introduction](https://pandas.pydata.org/docs/user_guide/categorical.html#categorical) and the [API documentation](https://pandas.pydata.org/docs/reference/arrays.html#api-arrays-categorical).

# In[172]:


df = pd.DataFrame(
    {"id": [1, 2, 3, 4, 5, 6], "raw_grade": ["a", "b", "b", "a", "a", "e"]}
)
df


# Converting the raw grades to a categorical data type:

# In[173]:


df["grade"] = df["raw_grade"].astype("category")
df["grade"]


# Rename the categories to more meaningful names:

# In[174]:


new_categories = ["very good", "good", "very bad"]
df["grade"] = df["grade"].cat.rename_categories(new_categories)
df


# Reorder the categories and simultaneously add the missing categories (methods under [`Series.cat()`](https://pandas.pydata.org/docs/reference/api/pandas.Series.cat.html#pandas.Series.cat) return a new [`Series`](https://pandas.pydata.org/docs/reference/api/pandas.Series.html#pandas.Series) by default):

# In[175]:


df["grade"] = df["grade"].cat.set_categories(
    ["very bad", "bad", "medium", "good", "very good"]
)
df["grade"]


# Sorting is per order in the categories, not lexical order:

# In[176]:


df.sort_values(by="grade")


# Grouping by a categorical column also shows empty categories:

# In[177]:


df.groupby("grade").size()


# ## Plotting
# 
# See the [Plotting](https://pandas.pydata.org/docs/user_guide/visualization.html#visualization) docs.
# 
# We use the standard convention for referencing the matplotlib API:

# In[178]:


import matplotlib.pyplot as plt
plt.close("all")

ts = pd.Series(np.random.randn(1000), index=pd.date_range("1/1/2000", periods=1000))
ts = ts.cumsum()
ts.plot();


# If running under Jupyter Notebook, the plot will appear on [`plot()`](https://pandas.pydata.org/docs/reference/api/pandas.Series.plot.html#pandas.Series.plot). Otherwise use [`matplotlib.pyplot.show`](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.show.html) to show it or [`matplotlib.pyplot.savefig`](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.savefig.html) to write it to a file.
# 
# On a DataFrame, the [`plot()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html#pandas.DataFrame.plot) method is a convenience to plot all of the columns with labels:

# In[179]:


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

# In[180]:


df.to_csv("foo.csv")


# [Reading from a csv file](https://pandas.pydata.org/docs/user_guide/io.html#io-read-csv-table): using [`read_csv()`](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html#pandas.read_csv)

# In[181]:


pd.read_csv("foo.csv")


# ### HDF5
# 
# Reading and writing to [HDFStores](https://pandas.pydata.org/docs/user_guide/io.html#io-hdf5).
# 
# Writing to a HDF5 Store using [`DataFrame.to_hdf()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_hdf.html#pandas.DataFrame.to_hdf):

# In[182]:


df.to_hdf("foo.h5", "df")


# Reading from a HDF5 Store using [`read_hdf()`](https://pandas.pydata.org/docs/reference/api/pandas.read_hdf.html#pandas.read_hdf):

# In[183]:


pd.read_hdf("foo.h5", "df")


# ### Excel
# 
# Reading and writing to [Excel](https://pandas.pydata.org/docs/user_guide/io.html#io-excel).
# 
# Writing to an excel file using [`DataFrame.to_excel()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_excel.html#pandas.DataFrame.to_excel):

# In[184]:


df.to_excel("foo.xlsx", sheet_name="Sheet1")


# Reading from an excel file using [`read_excel()`](https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html#pandas.read_excel):

# In[185]:


pd.read_excel("foo.xlsx", "Sheet1", index_col=None, na_values=["NA"])


# ## Gotchas
# 
# If you are attempting to perform a boolean operation on a [`Series`](https://pandas.pydata.org/docs/reference/api/pandas.Series.html#pandas.Series) or [`DataFrame`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html#pandas.DataFrame) you might see an exception like:

# In[186]:


if pd.Series([False, True, False]):
     print("I was true")


# See [Comparisons](https://pandas.pydata.org/docs/user_guide/basics.html#basics-compare) and [Gotchas](https://pandas.pydata.org/docs/user_guide/gotchas.html#gotchas) for an explanation and what to do.
