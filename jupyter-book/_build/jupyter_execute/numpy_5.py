#!/usr/bin/env python
# coding: utf-8

# (sec:numpy_5)=
# # 실전 예제: 어레이 활용

# 연산과 함수 호출에 사용되는 넘파이 어레이는 기본적으로 항목 단위로 연산과 함수 호출이 이루어진다.
# 넘파이 어레이의 이런 특징을 잘 활용하도록 유도하는 프로그래밍을 
# __어레이 중심 프로그래밍__(array-oriented programming)이라 한다. 

# **주요 내용**
# 
# - 예제: 2차원 격자 어레이
# - 예제: 붓꽃 데이터셋

# **기본 설정**
# 
# `numpy` 모듈과 시각화 도구 모듈인 `matplotlib.pyplot`에 대한 기본 설정을 지정한다.

# In[1]:


# 넘파이
import numpy as np

# 램덤 시드
np.random.seed(12345)

# 어레이 사용되는 부동소수점들의 정확도 지정
np.set_printoptions(precision=4, suppress=True)


# In[2]:


# 파이플롯
import matplotlib.pyplot as plt

# # 도표 크기 지정
# plt.rc('figure', figsize=(10, 6))


# **예제: 2차원 격자 어레이**

# 어레이를 중심으로 프로그래밍을 하면 예를 들어 많은 `for` 반복문을 생략할 수 있으며,
# 결과적으로 보다 효율적으로 코드를 구현할 수 있다.
# 또한 구현된 프로그램은 리스트를 이용하는 프로그램보다 빠르고 메모리 효율적으로 실행된다.
# 여기서는 몇 가지 예제를 이용하여 어레이 중심 프로그래밍을 소개한다. 

# 아래 모양의 격자무뉘에 해당하는 2차원 어레이를 생성하고자 한다.
# 각 점의 좌표는 -1과 1사이의 값을 20개의 구간으로 균등하게 나눈 값들이다. 
# 즉, 가로 세로 모두 21개의 점으로 구성된다.
# 
# __주의사항:__ `for` 반복문을 전혀 사용하지 않아야 한다.

# <div align="center" border="1px"><img src="https://github.com/codingalzi/pydata/blob/master/notebooks/images/graphs/meshgrid20x20.png?raw=true" style="width:400px;"></div>

# 먼저 `arange()` 함수를 이용하여 -1와 1 사이의 구간을 20개의 구간으로 균등하게 
# 나누는 어레이를 생성하려면
# 아래에서 처럼 -1에서 1.1 이전까지 0.1 스텝으로 증가하는 값들로 이루어진 어레이를 생성하면 된다.

# In[3]:


points = np.arange(-1, 1.1, 0.1) # -1부터 1.1 전까지 0.1 스텝으로 증가하는 값들의 어레이 생성

points


# **`np.meshgrid()` 함수**

# `meshgrid()` 함수는 지정된 1차원 어레이 두 개를 이용하여 격자무늬의 좌표를 생성한다.
# 즉, 격자에 사용되는 점들의 x 좌표와 y 좌표를 따로따로 모아 두 개의 어레이를 반환한다.

# In[4]:


xs, ys = np.meshgrid(points, points)


# In[5]:


xs


# In[6]:


ys


# xs와 ys를 이용하여 산점도를 그리면 원하는 격자무늬가 얻어진다. 

# In[7]:


# 도표 크기 지정
plt.rc('figure', figsize=(6, 6))

# 산점도 그리기
plt.scatter(xs, ys)
plt.show()


# __예제:__ 2차원 이미지 그리기

# xs와 ys 각각의 제곱을 합하여 제곱근을 구하면 21x21 크기의 대칭 어레이가 얻어진다. 

# In[8]:


z = np.sqrt(xs ** 2 + ys ** 2)


# In[9]:


z.shape


# In[10]:


z


# `z`를 흑백사진으로 표현하면 다음과 같다.
# `21x21` 크기의 해상도를 가진 흑백사진의 명암 대비를 쉽게 알아볼 수 있는 사진이 생성된다.

# In[11]:


# 도표 크기 지정(기본값으로 되돌림)
plt.rc('figure', figsize=(10, 6))

# 흑백사진으로 보여주기
plt.imshow(z, cmap=plt.cm.gray, extent=[-1, 1, 1, -1])

# 색막대(색상 지도): 수와 색 사이의 관계를 보여주는 일종의 색지도
plt.colorbar()

plt.show()


# __참고:__ 위 두 예제를 넘파이 어레이가 아니라 리스트와 `for` 반복문을 이용하여 구현하려고
# 시도하면 훨씬 많은 일을 해야 함을 어렵지 않게 알 수 있을 것이다.

# **예제**
# 
# -1부터 1사이의 구간은 0.02 크기로 총 100개의 구간으로 구성한 다음에 동일한 그래프를 그리면 훨씬 더 
# 섬세한 사진을 얻는다.

# In[12]:


points = np.arange(-1, 1.01, 0.02) # -1부터 1.1 전까지 0.02 스텝으로 증가하는 101 개의 값들로 이루어진 어레이 생성


# In[13]:


points.shape


# `meshgrid()` 함수를 이용하여 메쉬 생성에 필요한 x 좌표와 y 좌표 모음을 만든다.

# In[14]:


xs, ys = np.meshgrid(points, points)


# `xs` 와 `ys` 각각 (101, 101) 모양의 2차원 어레이다. 

# In[15]:


xs.shape


# In[16]:


ys.shape


# xs와 ys를 이용하여 산점도를 그리면 한 장의 색종이를 얻는다.
# 이유는 픽셀이 촘촘하기 때문이다. 

# In[17]:


# 도표 크기 지정
plt.rc('figure', figsize=(6, 6))

# 산점도 그리기
plt.scatter(xs, ys)
plt.show()


# 등고선 모양의 이미지를 생성하기 위해 xs와 ys 각각의 제곱을 합하여 제곱근을 구하면 101x101 모양의
# 2차원 대칭 어레이가 얻어진다. 

# In[18]:


z = np.sqrt(xs ** 2 + ys ** 2)


# In[19]:


z.shape


# `z`를 흑백사진으로 표현하면 다음과 같다.

# In[20]:


# 도표 크기 지정(기본값으로 되돌림)
plt.rc('figure', figsize=(10, 6))

# 흑백사진으로 보여주기
plt.imshow(z, cmap=plt.cm.gray, extent=[-1, 1, 1, -1])

# 색막대(색상 지도): 수와 색 사이의 관계를 보여주는 일종의 색지도
plt.colorbar()

plt.show()


# **예제: 붓꽃 데이터**

# 붓꽃(아이리스) 데이터를 이용하여 활용법을 살펴 보기 위해
# 먼저 데이터를 인터넷 상에서 가져온다. 

# In[21]:


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'


# 위 주소의 `iris.data` 파일을 `data`라는 하위 디렉토리에 저장한다.

# In[22]:


import os
import urllib.request

PATH = './data/'
os.makedirs(PATH, exist_ok=True)
urllib.request.urlretrieve(url, PATH+'iris.data')


# 다운로드된 `iris.data` 파일에는 아래 형식의 데이터가 150개 들어 있다. 
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

# In[23]:


get_ipython().system('cat data/iris.data | head -n 5')


# 이 중에 마지막 품종 특성은 문자열이고 나머지 특성은 부동소수점, 즉 수치형 데이터이다. 
# 여기서는 연습을 위해 수치형 데이터를 담고 있는 네 개의 특성만 가져온다.
# 
# * `genfromtxt()` 함수: 인터넷 또는 컴퓨터에 파일로 저장된 데이터를 적절한 모양의 어레이로 불러오는 함수
# * `delimiter=','`: 쉼표를 특성값들을 구분하는 기준으로 지정
# * `usecols=[0,1,2,3]`: 리스트에 지정된 인덱스의 특성만 가져오기

# In[24]:


iris_2d = np.genfromtxt(PATH+'iris.data', delimiter=',', dtype='float', usecols=[0,1,2,3])


# In[25]:


iris_2d.shape


# 처음 5개의 샘플은 앞서 살펴본 것과 동일하다.
# 이번에는 다만 2차원 어레이로 보일 뿐이다.

# In[26]:


iris_2d[:5]


# **문제** 
# 
# 2차원 어레이에서 결측치(`nan`)를 전혀 갖지 않은 행만 선택하는 함수 `drop_2d()`를 정의해보자. 

# **견본 답안**

# `iris_2d` 어레이를 이용하여 `drop_2d()` 함수를 어떻게 정의해야 할지 살펴보자.
# 먼저 `iris_2d` 어레이에 누락치의 존재 여부를 판단해야 한다.
# 
# `np.isnan()` 함수는 누락치가 있는 위치는 `True`, 나머지 위치는 `False`를 갖는 부울 어레이를 생성한다.

# In[27]:


np.isnan(iris_2d)[:5]


# 만약 결측치가 있다면 `True`가 한 번 이상 사용되었기에 `any()` 메서드를 이용하여 
# 누착치의 존재 여부를 판단할 수 있다.

# In[28]:


np.isnan(iris_2d).any()


# 그런데 누락치가 전혀 없다. 따라서 하나의 누락치를 임의로 만들어 보자.
# 예를 들어, 처음 5개 샘플의 꽃잎 너비(3번 열)의 값을 `nan`으로 대체하자.

# In[29]:


iris_2d[:5,3] = None


# In[30]:


iris_2d[:10]


# 이제 누락치가 존재하기에 `any()` 메서드는 `True`를 반환한다.

# In[31]:


np.isnan(iris_2d).any()


# `sum()` 함수를 이용하여 5개의 누락치가 있음을 정확하게 파악할 수도 있다. 
# 
# * `sum()` 함수: `True`는 1, `False`는 0으로 처리한다.

# In[32]:


np.sum(np.isnan(iris_2d))


# `sum()` 메서드를 사용할 수도 있다.

# In[33]:


np.isnan(iris_2d).sum()


# 행 단위로 누락치의 존재를 찾기 위해 행별로 `sum()` 함수를 실행한다. 
# 즉, 축을 1로 지정한다.

# In[34]:


np.sum(np.isnan(iris_2d), axis=1)[:10]


# 정확히 150개의 행에 대한 누락치 존재 여부를 보여준다.

# In[35]:


np.sum(np.isnan(iris_2d), axis=1).shape


# 이제 위 코드와 부울 인덱싱을 활용하여 누락치가 없는 행만 추출할 수 있다.

# In[36]:


mask = np.sum(np.isnan(iris_2d), axis=1) == 0


# In[37]:


iris_2d[mask].shape


# 위 어레이의 처음 5개의 샘플 데이터는 `iris_2d` 어레이에서 5번에서 9번 인덱스에 위치한 샘플 데이터와 동일하다.

# In[38]:


iris_2d[mask][:5]


# 이제 `drop_2d()` 함수를 다음과 같이 정의할 수 있다.

# In[39]:


def drop_2d(arr_2d):
    mask = np.isnan(arr_2d).sum(axis=1) == 0
    return arr_2d[mask]


# `iris_2d`에 위 함수를 적용하면 이전과 동일한 결과를 얻는다.

# In[40]:


drop_2d(iris_2d)[:5]


# **문제** 
# 
# iris_2d 데이터셋에 사용된 붓꽃들의 품종은 아래 세 개이다.

# In[41]:


a = np.array(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])


# 150개의 품종을 무작위로 선택하되 `Iris-setosa` 품종이 다른 품종들의 두 배로 선택되도록 하라.
# 
# 힌트: `np.random.choice()` 함수를 활용하라.

# **견본답안**

# `np.random.choice()` 함수의 `p` 키워드 인자를 이용한다.
# 사용되는 인자는 `[0.5, 0.25, 0.25]` 이다.

# In[42]:


np.random.seed(42)  # 무작위성 시드 지정
species_out = np.random.choice(a, 150, p=[0.5, 0.25, 0.25])


# 세 개의 이름 중에서 무작위로 150개의 이름을 선택하였다.

# In[43]:


species_out.shape


# 품종별 비율은 대략적으로 2:1:1 이다.

# In[44]:


setosa_ratio = (species_out == 'Iris-setosa').sum()/150
versicolor_ratio = (species_out == 'Iris-versicolor').sum()/150
virginica_ratio = (species_out == 'Iris-virginica').sum()/150

print(f"세토사, 버시컬러, 비르지니카 세 품종의 비율은 {setosa_ratio:.2f}:{versicolor_ratio:.2f}:{virginica_ratio:.2f} 이다.")


# ## 연습문제

# 참고: [(실습) 실전 예제: 어레이 활용](https://colab.research.google.com/github/codingalzi/datapy/blob/master/practices/practice-numpy_5.ipynb)
