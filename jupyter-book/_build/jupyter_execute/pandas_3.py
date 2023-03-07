#!/usr/bin/env python
# coding: utf-8

# (sec:pandas_3)=
# # 판다스 활용: 통계 기초

# **주요 내용**

# `Series`와 `DataFrame` 객체를로부터 기초 통계 자료를 추출하는 방식을 다룬다.
# 
# * 합, 평균, 표준편차
# * 상관관계, 공분산
# * 중복값 처리

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


# ## 합, 평균, 표준편차

# 기초 통계에서 사용되는 주요 메서드들의 활용법을 살펴본다.
# 
# * `sum()`
# * `mean()`
# * `std()`
# * `idxmax()`/`idxmin()`
# * `cumsum()`
# * `describe()`

# 기본적으로 열 단위로 작동하며, 결측치는 행 또는 열의 모든 값이 결측치가 아니라면 기본적으로 무시된다.
# 행 단위로 작동하게 하려면 축을 `axis=1` 또는 `axis='columns`로 지정하고,
# 결측치를 무시하지 않으려면 `skipna=False`로 지정한다.

# In[5]:


df = pd.DataFrame([[1.4, np.nan], [7.1, -4.5],
                   [np.nan, np.nan], [0.75, -1.3]],
                  index=['a', 'b', 'c', 'd'],
                  columns=['one', 'two'])
df


# * `sum()` 메서드: 행/열 단위 합 계산

# In[6]:


df.sum()


# 결측치를 무시하지 않으면, 결측치가 포함된 행/렬에 대한 계산은 하지 않는다.

# In[7]:


df.sum(skipna=False)


# In[8]:


df.sum(axis='columns')


# 시리즈는 하나의 열을 갖는 데이터프레임처럼 작동한다.

# In[9]:


df['one']


# In[10]:


df['one'].sum()


# * `mean()` 메서드: 평균값 계산

# In[11]:


df.mean()


# In[12]:


df.mean(axis='columns')


# 결측치를 무시하지 않으면, 결측치가 포함된 행/렬에 대한 계산은 하지 않는다.

# In[13]:


df.mean(skipna=False)


# In[14]:


df.mean(axis='columns', skipna=False)


# 시리즈의 경우도 동일하게 작동한다.

# In[15]:


df['one'].mean()


# In[16]:


df['one'].mean(skipna=False)


# * `std()` 메서드: 표준편차 계산

# In[17]:


df.std()


# In[18]:


df.std(axis='columns')


# In[19]:


df.std(skipna=False)


# In[20]:


df.std(axis='columns', skipna=False)


# * `idxmax()`/`idxmin()`: 최댓값/최솟값을 갖는 인덱스 확인

# 아래 코드는 열별 최댓값을 갖는 인덱스를 찾아준다.

# In[21]:


df.idxmax()


# 아래 코드는 행별 최솟값을 갖는 인덱스를 찾아준다.

# In[22]:


df.idxmin(axis=1)


# * `cumsum()`: 누적 합 계산

# In[23]:


df.cumsum()


# In[24]:


df.cumsum(skipna=False)


# * `describe()`: 요약 통계 보여주기

# 수치형 데이터의 경우 평균값, 표준편차, 사분위수 등의 통계 정보를 요약해서 보여준다.

# In[25]:


df.describe()


# 수치형 데이터가 아닐 경우 다른 요약 통계를 보여준다.

# In[26]:


ser = pd.Series(['a', 'a', 'b', 'c'] * 2)
ser


# In[27]:


ser.describe()


# ## 상관관계와 공분산

# 금융 사이트에서 구한 4 개 회사의 주가(price)와 거래량(volume)을 담고 있는 두 개의 데이터를 이용하여
# 상관계수와 공분산을 계산해본다.

# 이를 위해 먼저 바이너리 파일 두 개를 다운로드해서 지정된 하위 디렉토리에 저장한다.

# * 파일 저장 디렉토리 지정 및 생성

# In[28]:


from pathlib import Path

data_path = Path() / "examples"

data_path.mkdir(parents=True, exist_ok=True)


# * 특정 서버에서 파일 다운로드 함수

# In[29]:


import requests

# 파일 서버 기본 주소
base_url = "https://raw.githubusercontent.com/codingalzi/datapy/master/jupyter-book/examples/"
    
def myWget(filename):
    # 다운로드 대상 파일 경로
    file_url = base_url + filename
    
    # 저장 경로와 파일명
    target_path = data_path / filename

    data = requests.get(file_url)
    
    with open(target_path, 'wb') as f:
        f.write(data.content)


# 두 개의 픽클 파일 다운로드한다.
# - pkl 파일: 판다스에서 제공하는 객체를 `to_pickle()` 메서드를 이용하여 
#     컴퓨터에 파일로 저장할 때 사용되는 바이너리 파일.

# In[30]:


myWget("yahoo_price.pkl")


# In[31]:


myWget("yahoo_volume.pkl")


# 다운로드한 두 개의 데이터를 불러온다.
# 
# - `read_pickle()`: 저장된 pkl 파일을 파이썬으로 불러오는 함수

# 아래 코드는 일별 주가 데이터를 불러온다.
# 2010년 1월 4일부터 2016년 10월 21일까지의 데이터 1714개를 담고 있다.

# In[32]:


price = pd.read_pickle('examples/yahoo_price.pkl')
price


# 아래 코드는 동일 회사, 동일 날짜의 1일 거래량(volume) 담고 있는 데이터를 불러온다.

# In[33]:


volume = pd.read_pickle('examples/yahoo_volume.pkl')
volume


# 주가의 일단위 변화율을 알아보기 위해 퍼센트 변화율을 확인해보자.
# 
# __참고:__ 증권분야에서 return은 이익율을 의미한다.

# In[34]:


returns = price.pct_change()
returns.tail()


# **`corr()`/`cov()` 메서드**

# 상관계수와 공분산 모두 두 확률변수 사이의 선형관계를 보여주며
# 차이점은 다음과 같다.
# 
# - 공분산: 두 확률변수 $X, Y$ 사이의 선형관계를 계량화 함. 
#     양수/음수 여부에 따라 양 또는 음의 선형관계이며,
#     절댓값이 클 수록 강한 선형관계임.
#     다만, 사용되는 확률변수의 척도(scale)에 많은 영향을 받음.
#     따라서 보통 정규화한 값인 상관계수를 사용함.
# 
# $$
# \begin{align*}
# Cov(X, Y) & = E((X-\mu_X)(Y-\mu_Y))\\[2ex]
# \mu_X & = E(X) = \dfrac{\sum X}{n}\\[1.5ex]
# \mu_Y & = E(Y) = \dfrac{\sum Y}{n}
# \end{align*}
# $$
# 
# - 상관계수: 두 확률변수 사이의 선형관계를 -1과 1 사이의 값으로 표현.
#     양수/음수 여부에 따라 양 또는 음의 선형관계이며,
#     절댓값이 1에 가까울 수록 강한 선형관계임.
# $$
# \begin{align*}
# \rho & = \frac{Cov(X, Y)}{\sigma_X\cdot \sigma_Y}\\[2ex]
# \sigma_X & = \sqrt{Var(X)}\\[1.5ex]
# \sigma_X & = \sqrt{Var(X)}\\[1.5ex]
# Var(X) & = \dfrac{\sum (X-\mu_X)^2}{n}\\[1.5ex]
# Var(Y) & = \dfrac{\sum (X-\mu_Y)^2}{n}
# \end{align*}
# $$    

# 'MSFT'와 'IBM' 사이의 상관계수는 다음과 같다.

# In[35]:


returns['MSFT'].corr(returns['IBM'])


# 'MSFT'와 'IBM' 사이의 공분산은 다음과 같다.

# In[36]:


returns['MSFT'].cov(returns['IBM'])


# 아래와 같이 속성으로 처리할 수 있으면 보다 깔금하게 보인다.

# In[37]:


returns.MSFT.corr(returns.IBM)


# 전체 회사를 대상으로 하는 상관계수와 공분산을 계산할 수도 있다.

# In[38]:


returns.corr()


# In[39]:


returns.cov()


# **`corrwith()` 메서드: 다른 시리즈 또는 데이터프레임과의 상관계수 계산**

# 시리즈를 인자로 사용하면 각 열에 대한 상관계수를 계산한다.

# In[40]:


returns.corrwith(returns.IBM)


# 데이터프레임을 인자로 사용하면 공통 인덱스를 사용하는 모든 열에 대한 상관계수를 계산한다.

# In[41]:


returns.corrwith(volume)


# ## 중복과 빈도

# **`unique()` 메서드**

# 시리즈에서 사용된 값을 중복 없이 확인하려면 `unique()` 메서드를 이용한다.
# `set()` 함수와 유사하게 작동하며, 넘파이 어레이를 반환한다.

# In[42]:


obj = pd.Series(['c', 'a', 'd', 'a', 'a', 'b', 'b', 'c', 'c'])
obj


# In[43]:


uniques = obj.unique()
uniques


# **`value_counts()` 메서드**

# 값들의 빈도수를 확인하기 위해 사용한다.

# In[44]:


obj.value_counts()


# In[45]:


pd.value_counts(obj.values, sort=False)


# ## 미니 프로젝트: 붓꽃 데이터셋 표준화

# 아래 코드는 인터넷 데이터 저장소로부터 아이리스(붓꽃) 데이터(`iris.data`)를 
# 2차원 넘파이 어레이로 불러온다.

# In[46]:


# 아이리스(붓꽃) 데이터 불러오기
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='str')


# `iris.data` 파일에는 아래 형식의 데이터가 150개 들어 있다. 
# 
# ```python
# 5.1,3.5,1.4,0.2,Iris-setosa
# ```
# 
# 하나의 데이터에 사용된 값들은 하나의 아이리스(붓꽃)에 대한 꽃잎, 꽃받침과 관련된 특성(features)과 품종을 나타내며,
# 보다 구체적으로 아래 순서를 따른다.
# 
# ```
# 꽃받침 길이, 꽃받침 너비, 꽃잎 길이, 꽃잎 너비, 품종
# ```

# In[47]:


type(iris)


# In[48]:


iris.shape


# 길이와 너비를 저장하는 특성들은 숫자로 저장되어 있었지만 위 코드는 문자열로 저장된 품종 특성과의 자료형을 통일시키기 위해
# 모두 문자열 자료형으로 불러왔다.
# 처음 5개 데이터를 확인하면 다음과 같다.
# 
# __참고:__ `'<U15'`는 길이가 최대 15인 유니코드 문자열 자료형을 나타낸다.

# In[49]:


iris[:5]


# 수치형 데이터와 품종 데이터를 분리해서 각각 (150,4), (150,) 모양의 어레이를 생성하자.
# 이때 수치형 데이터는 `'f8'`, 즉 `'float64'` 자료형을 사용하도록 한다.

# In[50]:


iris_features = iris[:,:4].astype('f8')
iris_labels = iris[:, 4]


# 두 어레이를 판다스의 데이터프레임으로 형변환한다.
# 이때 각 열의 이름을 사용된 데이터 특성을 반영하도록 지정한다.

# In[51]:


columns = ['꽃받침길이', '꽃받침너비', '꽃잎길이', '꽃잎너비']
iris_features = pd.DataFrame(iris_features, columns=columns)
iris_features[:5]


# 레이블은 판다스의 시리즈로 변환한다.

# In[52]:


iris_labels = pd.Series(iris_labels)
iris_labels


# 150개의 데이터는 아래 세 개의 품종으로 구분되며, 각각 50개씩 아래 언급된 순서대로 구분되어 있다.
# 
# ```
# 'Iris-setosa', 'Iris-versicolor', 'Iris-virginica'
# ```
# 
# 즉, 0번, 50번, 100번부터 각 품종의 데이터가 시작된다.
# 넘파이의 경우와는 달리 인덱스를 항상 함께 보여준다.

# In[53]:


iris_labels[::50]


# In[54]:


iris_labels[:5]


# In[55]:


iris_labels[50:55]


# In[56]:


iris_labels[100:105]


# __예제 1.__ 꽃잎 길이(2번 열)가 1.5보다 크거나 꽃받침 길이(0번 열)가 5.0보다 작은 데이터만 추출하라.

# 부울 인덱싱은 넘파이의 경우와 기본적으로 동일하다.

# In[57]:


mask = (iris_features.꽃잎길이>1.5) | (iris_features.꽃받침길이<5.0)
mask


# 조건을 만족하는 샘플의 수는 아래와 같다.

# In[58]:


mask.sum()


# 조건을 만족하는 샘플들은 다음과 같다.
# 예를 들어 0번 4번 인덱스의 붓꽃은 조건에 맞지 않아 부울 인덱싱에서 걸러진다.
# 하지만 아래에서 볼 수 있듯이 다른 붓꽃의 인덱스가 조정되지는 않는다.
# 즉, 인덱스는 한 번 정해지면 절대 변하지 않는다.

# In[59]:


iris_features[mask]


# __예제 2.__ 꽃받침 길이(0번 열)와 꽃잎 길이(2번 열) 사이의 피어슨 상관계수를 계산하라.
# 
# 힌트: 넘파이의 적절한 함수를 활용한다. 상관계수에 대한 설명은 [위키백과: 상관분석](https://ko.wikipedia.org/wiki/상관_분석)을 참고한다.
# 상관계수를 구하는 여러 방식이 있지만 판다스의 `corr()` 함수는 피어슨 상관계수를 기본으로 계산한다.
# 다른 방식의 상관계수를 구하려며 `method` 키워드 인자를 다르게 지정해야 한다. 
# 보다 자세한 사항은 [판다스: `corr()` 함수 문서](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html)를 참조하라.

# 데이터프레임의 `corr()` 메서드는 모든 특성들 사이의 피어슨 상관계수로 이루어진 데이터프레임을 반환환다.

# In[60]:


iris_corr = iris_features.corr()
iris_corr


# 따라서 '꽃받침길이'와 다른 특성들 사이의 상관계수를 역순으로 정렬하면 다음과 같다.
# 
# - 인덱싱: 데이터프레임의 인덱싱은 기본적으로 열(column) 단위로 이루어진다. 
#     행(row) 단위 인덱싱은 `loc()` 또는 `iloc()` 메서드를 이용한다.
# - `sort_values()`: 열 단위로 오름차순으로 정렬함. 역순으로 하고자 할 경우 `ascending=False` 키워드 인자 사용.

# In[61]:


iris_corr['꽃받침길이'].sort_values(ascending=False)


# 따라서 '꽃받침길이'와 '꽃잎길이' 사이의 상관계수가 가장 높으며 
# 아래처럼 인덱싱을 두 번 사용하면 해당 값을 확인할 수 있다.

# In[62]:


iris_corr['꽃받침길이']['꽃잎길이']


# __예제 3.__ 아래 식으로 계산된 값을 갖는 새로운 열(column)이 추가된 데이터프레임 `iris_features_added`를 생성하라.
# 열의 이름은 '길이속성1'으로 지정한다.
# 
# $$\frac{\text{원주율} \times \text{꽃잎길이} \times \text{꽃받침길이}^2}{3} $$

# 시리즈를 생성하면서 동시에 `name='길이특성1'` 이라는 키워드 인자를 사용하는 이유는
# 이어서 `iris_features` 데이터프레임과 합칠 때 새로 추가되는 열의 이름으로 
# 사용되도록 하기 위함이다.

# In[63]:


# pass와 None을 각각 적절한 코드와 표현식으로 대체하라.

scaled = (3.14 * iris_features['꽃잎길이'] * iris_features['꽃받침길이']**2) / 3
length_property1 = pd.Series(scaled, name='길이특성1')


# In[64]:


length_property1


# In[65]:


iris_features_added = pd.concat([iris_features, length_property1], axis=1)

assert iris_features_added.shape == (150, 5)
iris_features_added


# __예제 4.__ `Iris_versicolor` 품종에 해당하는 데이터만 `iris_features`로부터 추출하라. 

# 부울 인덱싱을 사용한다.

# In[66]:


# None을 적절한 부울 표현식으로 대체하라.

mask = iris_labels == 'Iris-versicolor'
mask


# 정확히 50개의 샘플에 대해서만 `True`이다.

# In[67]:


mask.sum()


# 보다 정확히는 50번부터 99번까지 붓꽃만 선택된다.

# In[68]:


iris_versicolor = iris_features[mask]
iris_versicolor.head()


# In[69]:


iris_versicolor.tail()


# __예제 5.__ 꽃받침 길이(0번 열)의 평균값(mean), 중앙값(median), 표준편차(standard deviation)를 구하라.
# 
# __참고:__ 데이터프레임의 메서드는 기본적으로 열(columns)에 대한 속성을 다룬다.
# 즉, `axis=0`을 기본 축으로 사용한다.

# In[70]:


iris_mean = iris_features.mean()
iris_mean


# In[71]:


iris_mean = iris_features.mean(axis=0)
iris_mean


# In[72]:


iris_median = iris_features.median()
iris_median


# In[73]:


iris_std = iris_features.std()
iris_std


# 따라서 `for` 반복문을 이용하여 간단하게 세 개의 평균을 확인할 수 있다.
# 
# __참고:__ 특정 객체의 메소드로 이루어진 리스트에 포함된 메소드에 대한 반복문을 활용할 수 있다.
# 아래 코드는 통계와 관련해서 데이터프레임 객체가 제공하는 세 개의 메서드에 
# 대한 반복문을 적용하는 방식을 보여준다.

# In[74]:


average_methods = [pd.DataFrame.mean, pd.DataFrame.median, pd.DataFrame.std]

for fun in average_methods:
    print(fun(iris_features)['꽃받침길이'], end=' ')


# __예제 6.__ 세 개의 품종 별 꽃받침 너비(1번 열)의 평균값을 계산하여 아래 모양의  
# 데이터프레임과 시리즈(Series) `iris_mean_sepal_length`를 생성하라.
# 
# |                 | 평균 꽃받침 너비 |
# | ---:            | ---:             |
# | Iris-setosa     | 3.418            |
# | Iris-versicolor | 2.770            |
# | Iris-virginica  | 2.974            |
# 

# 데이터프레임을 만들려면 `index`와 `columns` 키워드를 인자를 적절하게 지정해야 한다.

# In[75]:


kinds = list(set(iris_labels))
kinds.sort()                      # 이름 순서를 맞추기 위해

iris_mean_sepal_width = []

for kind in kinds:
    mask = iris_labels == kind
    mean_0 = iris_features[mask].mean()['꽃받침너비']
    iris_mean_sepal_width.append(mean_0)
    
pd.DataFrame(iris_mean_sepal_width, index=kinds, columns=['평균 꽃받침 너비'])


# 시리즈를 만들려면 `index`와 `name` 키워드를 인자를 적절하게 지정해야 한다.

# In[76]:


kinds = list(set(iris_labels))
kinds.sort()                      # 이름 순서를 맞추기 위해

iris_mean_sepal_width = []

for kind in kinds:
    mask = iris_labels == kind
    mean_0 = iris_features[mask].mean()['꽃받침너비']
    iris_mean_sepal_width.append(mean_0)
    
pd.Series(iris_mean_sepal_width, index=kinds, name='평균 꽃받침 너비')


# __예제 7.__ 꽃잎 너비(3번 열)에 사용된 값을 모두 0과 1사이의 값으로 변환하라. 
# 
# 힌트: 하나의 특성, 여기서는 꽃잎 너비,에 속하는 값을 모두 0과 1사이의 값으로 변환하는 작업을 정규화(normalization)이라 한다.
# 정규화에 대한 설명은 [정규화/표준화](https://rucrazia.tistory.com/90)을 참고하라.

# In[77]:


iris_features[:5]


# 넘파이의 경우와 동일하게 작동한다.
# 하지만 데이터프레임의 메서드는 기본적으로 축을 0으로 지정해서 열 단위로 작동하기에
# 조금 더 간단하게 구현할 수 있다.

# In[78]:


iris_features.min()


# In[79]:


iris_features.min(axis=0)


# In[80]:


iris_features_normalized = (iris_features - iris_features.min())/(iris_features.max() - iris_features.min())

iris_features_normalized


# 이제 꽃잎 너비에 대한 정보만 인덱싱으로 추출하면 된다.

# In[81]:


iris_features_normalized.꽃잎너비


# __예제 8.__ `iris_features`에 사용된 모든 값을 특성 별로 표준화(standardization)하라. 
# 
# 힌트: 표준화에 대한 설명은 [정규화/표준화](https://rucrazia.tistory.com/90)을 참고하라.

# `mean()`, `std()` 메서드도 기본적으로 축을 0으로 지정해서 작동한다.

# In[82]:


iris_features.mean()


# In[83]:


iris_features.std()


# In[84]:


# None을 적절한 부울 표현식으로 대체하라.

iris_features_standardized = (iris_features - iris_features.mean()) / iris_features.std()

iris_features_standardized[:5]


# ## 연습문제 

# 참고: [(실습) 판다스 활용: 통계 기초](https://colab.research.google.com/github/codingalzi/datapy/blob/master/practices/practice-pandas_stats.ipynb)
