#!/usr/bin/env python
# coding: utf-8

# (sec:numpy_5)=
# # 실전 예제: 어레이 활용

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


# ## 2차원 격자 어레이

# 아래 모양의 회색 격자무늬에 해당하는 2차원 어레이를 생성하고자 한다.
# 각 점의 좌표는 -1과 1사이의 값을 10개의 구간으로 균등하게 나눈 값들이다. 
# 즉, 가로 세로 모두 11개의 점으로 구성된다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/datapy/master/jupyter-book//images/meshgrid10x10.png" style="width:350px;"></div>

# 먼저 `np.arange()` 함수를 이용하여 -1와 1 사이의 구간을 10개의 구간으로 균등하게 
# 나누는 어레이를 생성하려면
# 아래에서 처럼 -1에서 1.01 이전까지 0.2 스텝으로 증가하는 값들로 이루어진 어레이를 생성하면 된다.

# In[3]:


points = np.arange(-1, 1.01, 0.2)

points


# **`np.meshgrid()` 함수**

# 예를 들어 `matplotlib.pyplot` 모듈의 `scatter()` 함수를 이용하여 위 그림에 있는 총 121(= 11 $\times$ 11)개의 점을
# 산점도로 그리려면 각각 121개의 x-좌표와 y-좌표를 담은 두 개의 리스트가 필요하다.
# 그리고 `np.meshgrid()` 함수를 이용하면 손쉽게 두 리스트를 구할 수 있다.
# 
# `np.meshgrid()` 함수는 지정된 1차원 어레이 두 개를 이용하여 그릴 수 있는 격자무늬의 
# x-좌표 리스트와 y-좌표 리스트를 생성하며
# 아래와 같이 실행한다.

# In[4]:


xs, ys = np.meshgrid(points, points)


# `xs`는 열별x-좌표를 2차원 어레이로 담고 있다.
# 열 순서는 왼쪽에서 오른쪽으로 진행한다.

# In[5]:


xs


# `ys`는 행별 y-좌표를 2차원 어레이로 담고 있다.
# 행 순서는 아래에서 위쪽으로 진행한다.

# In[6]:


ys


# xs와 ys를 이용하여 산점도를 그리면 원하는 격자무늬가 얻어진다. 

# In[7]:


# 도표 크기 지정
plt.rc('figure', figsize=(5, 5))

# 산점도 그리기
plt.scatter(xs, ys, c='darkgray')
plt.show()


# **배경화면 그리기**

# 회색 배경화면을 얻고자 한다면 보다 점을 보다 촘촘히 찍으면 된다.
# 예를 들어, -1부터 1사이의 구간은 0.02 크기로 총 100개의 구간으로 구성한 다음에 동일한 그래프를 그리면 훨씬 더 
# 섬세한 사진을 얻는다.

# In[8]:


points = np.arange(-1, 1.01, 0.02)


# `meshgrid()` 함수를 이용하여 메쉬 생성에 필요한 x 좌표와 y 좌표 모음을 만든다.

# In[9]:


xs, ys = np.meshgrid(points, points)


# `xs` 와 `ys` 각각 (101, 101) 모양의 2차원 어레이다. 

# In[10]:


xs.shape


# In[11]:


ys.shape


# xs와 ys를 이용하여 산점도를 그리면 한 장의 회색 색종이를 얻는다.
# 이유는 픽셀이 촘촘하기 때문이다. 

# In[12]:


# 도표 크기 지정
plt.rc('figure', figsize=(6, 6))

# 산점도 그리기
plt.scatter(xs, ys, c= 'darkgray')
plt.show()


# 등고선 모양의 이미지를 생성하기 위해 xs와 ys 각각의 제곱을 합하여 제곱근을 구하면 101x101 모양의
# 2차원 대칭 어레이가 얻어진다. 

# In[13]:


z = np.sqrt(xs ** 2 + ys ** 2)


# In[14]:


z.shape


# `z` 값을 기준으로 등고선을 흑백사진으로 표현하면 다음과 같다.

# In[15]:


# 도표 크기 지정(기본값으로 되돌림)
plt.rc('figure', figsize=(6, 6))

plt.contourf(xs, ys, z, cmap=plt.cm.gray)

plt.show()


# ## 붓꽃 데이터셋

# 붓꽃(아이리스) 데이터를 이용하여 활용법을 살펴 보기 위해
# 먼저 데이터를 인터넷 상에서 가져온다. 

# In[16]:


url = 'https://raw.githubusercontent.com/codingalzi/datapy/master/jupyter-book/data/iris_nan.data'
# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'


# 위 주소의 `iris.data` 파일을 `data`라는 하위 디렉토리에 저장한다.

# In[17]:


from pathlib import Path
import urllib.request

data_path = Path() / "data"
data_path.mkdir(parents=True, exist_ok=True)
urllib.request.urlretrieve(url, data_path / 'iris.data')


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

# 이 중에 마지막 품종 특성은 문자열이고 나머지 특성은 부동소수점, 즉 수치형 데이터이다. 
# 여기서는 연습을 위해 수치형 데이터를 담고 있는 네 개의 특성만 가져온다.
# 
# * `genfromtxt()` 함수: 인터넷 또는 컴퓨터에 파일로 저장된 데이터를 적절한 모양의 어레이로 불러오는 함수
# * `delimiter=','`: 쉼표를 특성값들을 구분하는 기준으로 지정
# * `usecols=[0,1,2,3]`: 리스트에 지정된 인덱스의 특성만 가져오기

# In[18]:


iris_2d = np.genfromtxt(data_path / 'iris.data', delimiter=',', dtype='float', usecols=[0,1,2,3])


# 어레이의 모양은 (150, 4)이다. 

# In[19]:


iris_2d.shape


# 처음 5개 샘플은 다음과 같다.

# In[20]:


iris_2d[:5]


# **결측치 처리** 

# 붓꽃 데이터셋 안에 결측치가 포함되어 있다.
# 누라치가 있는지 여부를 다음과 같이 확인한다.
# 
# - `np.isnan()` 함수: 어레이의 각각의 항목이 결측치인지 여부를 확인하는 부울 어레이 반환
# - `any()` 어레이 메서드: 부울 어레이의 항목에 `True`가 하나라도 포함되어 있는지 여부 확인

# In[21]:


np.isnan(iris_2d).any()


# 결측치가 특정 열에만 있는지를 확인하려면 축을 0으로 지정한다.

# In[22]:


np.isnan(iris_2d).any(axis=0)


# 3번 열에만 결측치가 있음이 확인됐다.

# `sum()` 함수를 이용하여 3개의 누락치가 있음을 바로 확인할 수 있다.
# 
# * `sum()` 어레이 메서드: `True`는 1, `False`는 0으로 처리한다.

# In[23]:


np.isnan(iris_2d).sum()


# 3번 열에만 결측치가 있기에 아래와 같이 결측치의 수를 확인할 수도 있다.

# In[24]:


np.isnan(iris_2d[:, 3]).sum()


# 부울 인덱싱을 활용하여 누락치가 없는 행만 추출할 수 있다.

# In[25]:


mask = np.isnan(iris_2d[:, 3])


# 147개의 행에는 누락치가 없다.

# In[26]:


iris_2d[~mask].shape


# 누락치를 포함한 데이터 샘플 3개는 다음과 같다.

# In[27]:


iris_2d[mask]


# `nan`은 결측치를 의미하는 값인 `np.nan`을 가리키는 기호다.

# In[28]:


np.nan


# 3개의 결측치는 사실 일부러 만들어졌고 원래 모두 0.2였다.
# 따라서 결측치를 모두 0.2로 바꾸고 다음 과정을 실행한다.

# In[29]:


iris_2d[:, 3][mask] = 0.2


# 결측치가 없음을 다음과 같이 확인한다.

# In[30]:


np.isnan(iris_2d).any()


# **문제** 
# 
# iris_2d 데이터셋에 사용된 붓꽃들의 품종은 아래 세 개이다.

# In[31]:


a = np.array(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])


# 150개의 품종을 무작위로 선택하되 `Iris-setosa` 품종이 다른 품종들의 두 배로 선택되도록 하라.
# 
# 힌트: `np.random.choice()` 함수를 활용하라.

# **견본답안**

# `np.random.choice()` 함수의 `p` 키워드 인자를 이용한다.
# 사용되는 인자는 `[0.5, 0.25, 0.25]` 이다.

# In[32]:


np.random.seed(42)  # 무작위성 시드 지정
species_out = np.random.choice(a, 150, p=[0.5, 0.25, 0.25])


# 세 개의 이름 중에서 무작위로 150개의 이름을 선택하였다.

# In[33]:


species_out.shape


# 품종별 비율은 대략적으로 2:1:1 이다.

# In[34]:


setosa_ratio = (species_out == 'Iris-setosa').sum()/150
versicolor_ratio = (species_out == 'Iris-versicolor').sum()/150
virginica_ratio = (species_out == 'Iris-virginica').sum()/150

print(f"세토사, 버시컬러, 비르지니카 세 품종의 비율은 {setosa_ratio:.2f}:{versicolor_ratio:.2f}:{virginica_ratio:.2f} 이다.")


# ## 연습문제

# 참고: [(실습) 실전 예제: 어레이 활용](https://colab.research.google.com/github/codingalzi/datapy/blob/master/practices/practice-numpy_5.ipynb)
