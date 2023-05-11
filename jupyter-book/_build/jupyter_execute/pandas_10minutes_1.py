#!/usr/bin/env python
# coding: utf-8

# (sec:pandas10min_1)=
# # 판다스 10분 완성 1부
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

# In[17]:


df["A"]


# 객체의 속성처럼 이용하는 방식도 가능.

# In[18]:


df.A


# ### 행 슬라이싱

# 정수 인덱스 활용. 데이터프레임 생성

# In[19]:


df[0:3]


# 인덱스 라벨 활용. 정수 인덱스 방식과 조금 다름.

# In[20]:


df["20130101":"20130103"]


# ### `loc[]`: 라벨 활용 인덱싱/슬라이싱

# 인덱스 라벨을 이용하면 열 라벨을 인덱스로 사용하는 시리즈가 생성된다.

# In[21]:


dates[0]


# In[22]:


df.loc[dates[0]]


# 축 활용. 행과 열에 대한 인덱싱/슬라이싱 동시에 지정.

# - `A`, `B` 두 열만 추출.

# In[23]:


df.loc[:, ["A", "B"]]


# - 특정 행만 대상으로 `A`, `B` 두 열 추출

# In[24]:


df.loc["20130102":"20130104", ["A", "B"]]


# 인덱싱이 사용되면 차원이 줄어듦.

# In[25]:


df.loc["20130102", ["A", "B"]]


# 두 개의 인덱싱은 결국 하나의 상수(스칼라) 생성.

# In[26]:


df.loc[dates[0], "A"]


# 하나의 항목을 선택할 때 `at` 함수 사용.

# In[27]:


df.at[dates[0], "A"]


# ### `iloc[]`: 정수 인덱스 활용 인뎅식/슬라이싱

# 행 선택

# In[28]:


df.iloc[3]


# 어레이 인덱싱/슬라이싱 방식

# In[29]:


df.iloc[3:5, 0:2]


# 넘파이 어레이의 팬시 인덱싱과는 다르게 작동한다.

# In[30]:


df.iloc[[1, 2, 4], [0, 2]]


# In[31]:


df.iloc[[1, 2, 4], [0, 2, 3]]


# In[32]:


df.iloc[[1, 2, 4], [0, 2, 3, 1]]


# 행 슬라이싱

# In[33]:


df.iloc[1::2, :]


# 열 슬라이싱

# In[34]:


df.iloc[:, 1:3]


# 하나의 항목 추출

# In[35]:


df.iloc[1, 1]


# `iat[]` 활용도 가능

# In[36]:


df.iat[1, 1]


# ### 부울 인덱싱

# 마스크 활용

# - `A` 열에 양수 항목이 있는 행만 추출

# In[37]:


df["A"] > 0


# In[38]:


df[df["A"] > 0]


# 양수 항목만 그대로 두고 나머지는 결측치로 처리

# In[39]:


df[df > 0]


# 넘파이 어레이 방식과 다르게 작동한다.
# 아래 코드에서처럼 양수 항목만 모은 1차원 어레이가 생성된다.

# In[40]:


aArray = df.to_numpy()
aArray


# In[41]:


aArray[aArray > 0]


# - [`isin()` 메서드](https://pandas.pydata.org/docs/reference/api/pandas.Series.isin.html#pandas.Series.isin) 활용

# In[42]:


df2 = df.copy()
df2["E"] = ["one", "one", "two", "three", "four", "three"]
df2


# `E` 열에 `"two"` 또는 `"four"` 가 항목으로 사용된 행만 `True`

# In[43]:


df2["E"].isin(["two", "four"])


# `E` 열에 `"two"` 또는 `"four"` 가 항목으로 사용된 행만 추출하기

# In[44]:


df2[df2["E"].isin(["two", "four"])]


# ### 항목 지정

# In[45]:


s1 = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range("20130102", periods=6))
s1


# `F` 열 추가. 항목은 `s1` 이용.
# 0번 행은 결측치로 처리됨.

# In[46]:


df["F"] = s1
df


# `at[]` 활용: 첫재 행, `A` 열 항목을 0으로 지정.

# In[47]:


df.at[dates[0], "A"] = 0

df


# `iat[]`도 활용 가능.

# In[48]:


df.iat[0, 1] = 0


# 어레이를 이용하여 열 또는 행을 지정할 수 있다.

# - `D` 열 항목 지정

# In[49]:


df.loc[:, "D"] = np.array([5] * len(df))


# In[50]:


df


# - 1번 행 항목 지정

# In[51]:


df.loc[dates[1], :] = np.array([3] * df.shape[1])


# In[52]:


df


# - `iloc[]` 도 사용 가능

# In[53]:


df.iloc[2, :] = np.array([4] * df.shape[1])


# In[54]:


df


# ### `where()`/`mask()` 메서드 활용

# 참고
# 
# - [`DataFrame.where()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.where.html)
# - [`DataFrame.mask()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.mask.html)

# In[55]:


s = pd.Series(range(5))


# `where(조건식)`은 시리즈/데이터프레임의 항목 중에서 조건식이
# 거짓이 되도록 하는 항목 모두 결측치로 처리한 시리즈/데이터프레임을 생성한다.

# In[56]:


s.where(s > 0)


# In[57]:


df.where(df > 0)


# `mask(조건식)`은 시리즈/데이터프레임의 항목 중에서 조건식이
# 참이 되도록 하는 항목 모두 결측치로 처리한 시리즈/데이터프레임을 생성한다.

# In[58]:


s.mask(s > 0)


# In[59]:


s.mask(s <= 0)


# In[60]:


df.mask(df > 0)


# In[61]:


df.mask(df <= 0)


# `where()`/`mask()`의 인자로 부울 시리즈 또는 부울 데이터프레임이 사용될 수 있다.
# 그러면 `True`가 위치한 곳만 대상으로 마스크가 작동한다.

# In[62]:


t = pd.Series([True, False, False, True])

t


# 0번, 3번 위치만 참으로 처리된다.
# 4번 위치처럼 마스크에서 아예 위치로 언급되지 않는 경우는 무조건 거짓으로 처리된다.

# In[63]:


s.where(t)


# `mask()` 메서드는 4번 위치처럼 마스크에서 언급되지 않은 곳은 무조건 참으로 처리한다.

# In[64]:


s.mask(t)


# In[65]:


df2 = df.copy()
df2


# 연습을 위해 결측치를 제거한다.

# In[66]:


df2.iloc[0, -1] = 4.
df2


# `where()`/`mask()`가 두 개의 인자를 사용하면
# 조건식이 참/거짓이 되는 항목을 결측치가 아닌 둘째 인자로 대체한다.

# - 양수 항목은 해당 값의 음수로 대체

# In[67]:


df3 = df2.where(df2 > 0, -df2)
df3


# 아래처럼 부울 인덱싱하는 것과 동일하다.

# In[68]:


df4 = df2.copy()
df4[df2 <= 0] = -df2
df4


# In[69]:


(df3 == df4).all(axis=None)


# `mask()` 메서드도 유사하게 작동한다.

# In[70]:


df5 = df2.mask(df2 <= 0, -df2)


# In[71]:


(df3 == df5).all(None)  # axis=None


# ## 결측치

# - 내부적으로 `np.nan`을 사용. 겉으로는 자료형에 따라 
#     `NaN`(부동소수점), `NA`(정수), `NaT`(시간) 등으로 표기.
# - 참고: [Missing Data](https://pandas.pydata.org/docs/user_guide/missing_data.html#missing-data)

# 결측치가 포함된 어떤 연산도 결측치로 처리된다.

# In[72]:


np.nan + 1


# 심지어 두 결측치의 비교도  허영 안된다.

# In[73]:


np.nan == np.nan


# 반면에 `None`은 하나의 값으로 간주되어 비교가 가능하다.

# In[74]:


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

# In[75]:


df


# `reindex()` 메서드를 이용하여 행과 열의 라벨을 새로 지정한다.

# In[76]:


df1 = df.reindex(index = dates[0:4], columns = list(df.columns) + ['E'])
df1


# 결측치 일부를 채운다.

# In[77]:


df1.loc[dates[0] : dates[1], 'E'] = 1
df1


# - [`DataFrame.dropna()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html#pandas.DataFrame.dropna) 메서드: 결측치를 포함한 행을 삭제한 데이터프레임 생성
# 
# 

# In[78]:


df1.dropna(how='any')


# - [`DataFrame.fillna()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.fillna.html#pandas.DataFrame.fillna): 결측치를 지정된 값으로 채운 데이터프레임 생성

# In[79]:


df1.fillna(value=5)


# - [`isna()`](https://pandas.pydata.org/docs/reference/api/pandas.isna.html#pandas.isna):
#     결측치가 위치한 곳만 `True`로 처리하는 부울 마스크 생성

# In[80]:


pd.isna(df)


# ## 연산

# - 참고: [Basic section on Binary Ops](https://pandas.pydata.org/docs/user_guide/basics.html#basics-binop).

# ### 통계

# 주의사항
# 
# - 결측치는 무시된다.

# In[81]:


df


# `F` 열은 결측치를 제외한 5개의 값의 평균값을 구한다.

# In[82]:


df.mean()


# 실제로 `F` 열에서 결측치를 제외한 항목 개수는 5이다.

# In[83]:


df.F.value_counts()


# In[84]:


df.F.value_counts().sum()


# 결측치를 제외한 항목의 합을 5로 나눈 값은 3.8이다.

# In[85]:


df.F.sum()/5


# **축 활용**
# 
# 축을 지정하면 행 또는 열 기준으로 작동한다.

# - 행별 평균값

# In[86]:


df.mean(1) # axis=1


# ### 사칙연산

# In[87]:


s = pd.Series([1, 3, 5, np.nan, 6, 8], index = dates)
s


# 결측치를 더 추가한다.
# 
# 참고; [`DataFrame.shift()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.shift.html)

# In[88]:


s = pd.Series([1, 3, 5, np.nan, 6, 8], index = dates).shift(2)
s


# - 행별 뺄셈: `df - s`

# In[89]:


df


# 결츠치가 관여하면 무조건 결측치로 처리된다.

# In[90]:


df.sub(s, axis='index') # axis=0


# 브로드캐스팅은 필요에 따라 자동 적용된다.

# In[91]:


df - 5


# ### 함수 적용

# - 참고: [`DataFrame.apply()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.apply.html#pandas.DataFrame.apply)

# In[92]:


df


# - 열별 누적합

# In[93]:


df.apply(np.cumsum)


# - 행별 누적합

# In[94]:


df.apply(np.cumsum, axis='columns') # axis=1


# - 열별 최대값과 최소값의 차이. 결측치는 무시

# In[95]:


df.apply(lambda x: x.max() - x.min())


# - 행별 최대값과 최소값의 차이. 결측치는 무시

# In[96]:


df.apply(lambda x: x.max() - x.min(), axis=1)


# ### 이산화

# 참고: [Histogramming and Discretization](https://pandas.pydata.org/docs/user_guide/basics.html#basics-discretization)

# In[97]:


np.random.seed(17)

arr = pd.Series(np.random.randn(20))
arr


# `hist()` 메서드는 값의 범위를 10등분해서 각 구간에 속한 값들의 개수를 히스토그램으로 보여준다.

# In[98]:


arr.hist() # bins=10 이 기본


# 전체 값의 범위를 4등분한 다음에 막대그래프를 그려보자.

# In[99]:


factor = pd.cut(arr, bins=4)
factor


# 구간별 항목의 개수 확인

# In[100]:


factor.value_counts()


# 막대그래프 그리기

# In[101]:


factor.value_counts().sort_index().plot.bar(rot=0, grid=True)


# 4등분한 구간에 라벨을 붙이면 정보를 보다 정확히 전달한다.

# In[102]:


factor = pd.cut(arr, bins=4, labels=['A', 'B', 'C', 'D'])
factor


# In[103]:


factor.value_counts().sort_index().plot.bar(rot=0, grid=True)


# ### 문자열 메서드 활용

# 참고
# 
# - [정규식](https://docs.python.org/3/library/re.html)
# - [벡터와 문자열 메서드](https://pandas.pydata.org/docs/user_guide/text.html#text-string-methods)

# `str` 속성은 모든 항목을 문자열로 변환한 벡터를 가리킨다.

# In[104]:


s = pd.Series(["A", "B", "C", "Aaba", "Baca", np.nan, "CABA", "dog", "cat"])
s.str


# 변환된 벡터에 문자열 메서드를 적용하면 새로운 시리즈가 생성된다.

# In[105]:


s.str.lower()


# In[106]:


df = pd.DataFrame(np.random.randn(3, 2), columns=[" Column A ", " Column B "], index=range(3))
df


# 열 라벨에 대해 문자열 메서드를 적용해보자.

# - 소문자화

# In[107]:


df.columns.str.lower()


# - 양끝의 공백 제거

# In[108]:


df.columns.str.lower().str.strip()


# - 중간에 위치한 공백을 밑줄(underscore)로 대체

# In[109]:


df.columns.str.strip().str.lower().str.replace(" ", "_")

