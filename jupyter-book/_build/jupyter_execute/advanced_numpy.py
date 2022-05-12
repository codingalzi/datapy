#!/usr/bin/env python
# coding: utf-8

# (ch:advanced_numpy)=
# # 고급 넘파이

# 넘파이 어레이를 조작하는 고급 기능 일부를 예제를 이용하여 살펴본다.

# **주요 내용**
# 
# * 어레이 모양 변경
# * 어레이 쪼개기/쌓기
# * 브로드캐스팅

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

# 파이플롯
import matplotlib.pyplot as plt
# 도표 크기 지정
plt.rc('figure', figsize=(10, 6))


# ## 어레이 조작 고급 기법

# 어레이를 조작하는 몇 가지 중요한 고급 기법을 살펴본다. 

# ### 어레이 모양 변형

# **`reshape()` 메서드**

# `reshape()` 메서드를 활용하여 지정된 튜플의 모양으로 주어진 어레이의 모양을 변형한다.
# 단, 항목의 수가 변하지 않도록 모양을 지정해야 한다.
# 
# 예를 들어, 길이가 8인 1차원 어레이가 다음과 같다.

# In[2]:


arr = np.arange(8)
arr


# 이제 (4, 2) 모양의 2차원 어레이로 모양을 변형할 수 있다.

# In[3]:


arr.reshape((4, 2))


# 항목의 수만 같으면 임의의 차원의 어레이를 임의의 차원의 어레이로 변형시킬 수 있다.

# In[4]:


arr.reshape((4, 2)).reshape((2, 2, 2))


# **`-1`의 역할**

# 어레의 모양을 지정할 때 튜플의 특정 위치에 -1을 사용할 수 있다.
# 그러면 그 위치의 값은 튜플의 다른 항목의 정보를 이용하여 자동으로 지정된다.
# 예를 들어, 아래 코드에서 -1은 4를 의미한다. 
# 이유는 20개의 항목을 5개의 행으로 이루어진 2차원 어레이로 지정하려면 열은 4개 있어야 하기 때문이다.

# In[5]:


arr = np.arange(20)
arr.reshape((5, -1))


# In[6]:


arr.reshape((5, 4))


# 동일한 이유로 아래에서 -1은 5를 의미한다.

# In[7]:


arr.reshape((2, -1, 2))


# In[8]:


arr.reshape((2, 5, 2))


# ### 차원 추가

# 어레이에 임의의 축을 추가하는 방식으로 1차원 커진 어레이를 생성할 수 있다.
# 어느 축을 지정하느냐에 따라 다른 모양을 갖게 된다.

# **예제**

# 다음 길이가 3인 1차원 어레이를 이용하자.
# 
# * `np.random.normal()` 함수는 `np.random.randn()` 함수를 일반화하여 정균분포를 따르면서
#     무작위 수를 생성한다. 
#     평균값과 표준편차를 지정할 수 있으며, 기본값은 평균값 0, 표준편차 1로 표준 정규분포를 따르도록 한다.

# In[9]:


arr_1d = np.random.normal(size=3)
arr_1d


# 아래 코드는 열 관련 축을 추가한다.

# In[10]:


arr_1d[:, np.newaxis]


# `reshape()` 메소드로도 동일한 결과를 얻을 수 있다.

# In[11]:


arr_1d.reshape((3, 1))


# 아래 코드는 행 관련 축을 추가한다.

# In[12]:


arr_1d[np.newaxis, :]


# In[13]:


arr_1d.reshape((1,3))


# **예제**

# 2차원 어레이에 축을 추가하면 3차원 어레이가 생성되며, 작동방식은 앞서와 동일하다.

# In[14]:


arr = np.random.randn(4, 3)
arr


# In[15]:


arr[:,:,np.newaxis].shape


# In[16]:


arr[:,:,np.newaxis]


# In[17]:


arr[:,np.newaxis,:].shape


# In[18]:


arr[:,np.newaxis,:]


# **`ravel()` 메서드와 `flatten()` 메서드**

# 두 메서드 모두 1차원 어레이를 반환한다. 
# 즉, 차원을 모두 없앤다.

# In[19]:


arr = np.arange(15).reshape((5, 3))
arr


# In[20]:


arr1 = arr.ravel()
arr1


# In[21]:


arr2 = arr.flatten()
arr2


# 차이점은 `ravel()` 메서드는 뷰(view)를 사용하는 반면에 `flatten()` 메서드는 어레이를 새로 생성한다.
# 예를 들어, 아래처럼 `arr1`의 항목을 변경하면 `arr`의 항목도 함께 변경된다.

# In[22]:


arr1[0] = -1
arr


# `arr2`은 `arr`과 전혀 상관이 없다.

# In[23]:


arr2[0] = -7
arr


# ### 어레이 쪼개기/쌓기

# **`np.split()` 함수**

# 어레이를 지정된 기준에 따라 여러 개의 어레이로 쪼갠다.
# 반환값은 쪼개진 어레이들의 리스트다.
# 
# 아래 예제를 살펴보자.

# In[24]:


arr = np.random.randn(7, 5)
arr


# `np.split()` 함수의 인자는 정수이거나 정수들의 리스트가 사용된다.
# 먼저, 정수 리스트가 들어오면 축이 정한 방향으로 리스트에 포함된 정수를 이용하여 여러 개의 구간으로 쪼갠다.
# 
# 아래 코드는 행을 기준으로 행의 인덱스를 0-1, 2, 3-4, 5-7 네 개의 구간으로 쪼갠다.
# 따라서 결과는 네 개의 어레이로 이루어진 리스트가 되며,
# 각 어레의 모양은 다음과 같다.
# 
# ```python
# (2, 5), (1, 5), (2, 5), (2, 5)
# ```

# In[25]:


np.split(arr, [2, 3, 5],axis=0)


# 반면에 열을 기준으로 0, 1-2, 3-4 3개의 구간으로 쪼개면 다음과 같으며,
# 각 어레이의 모양은 다음과 같다.
# 
# ```python
# (7, 1) (7, 2), (7, 2)
# ```

# In[26]:


np.split(arr, [1, 3],axis=1)


# **`np.vsplit()`/`np.hsplit()` 함수**

# 두 함수는 `np.split()` 함수에 축을 각각 0과 1로 지정한 함수이다.

# * `np.vsplit(arr, z)` = `np.split(arr, z, axis=0)`

# In[27]:


np.vsplit(arr, [2, 3, 5])


# * `np.hsplit(arr, z)` = `np.split(arr, z, axis=1)`

# In[28]:


np.hsplit(arr, [1, 3])


# **`np.concatenate()` 함수**

# 두 개의 어레이를 이어 붙이다.
# 지정되는 축에 따라 좌우로 또는 상하로 이어붙인다.
# 
# 아래 두 어레이가 주어졌다고 가정하다.

# In[29]:


arr1 = np.array([[1, 2, 3], [4, 5, 6]])
arr1


# In[30]:


arr2 = np.array([[7, 8, 9], [10, 11, 12]])
arr2


# 위아래로 이어붙이려면 축을 0으로 정한다.
# 
# **주의사항:** 인자가 길이가 2인 리스트이다.

# In[31]:


np.concatenate([arr1, arr2], axis=0)


# 좌우로 이어붙이려면 축을 1로 정한다.

# In[32]:


np.concatenate([arr1, arr2], axis=1)


# **`np.vstack()`/`np.hstack()` 함수**

# 두 함수는 `np.concatenate()` 함수에 축을 각각 0과 1로 지정한 함수이다.
# 
# **주의사항:** 인자가 길이가 2인 튜플이다.

# * `np.vstack((x, y))` = `np.concatenate([x, y], axis=0)`

# In[33]:


np.vstack((arr1, arr2))


# * `np.hstack((x, y))` = `np.concatenate([x, y], axis=1)`

# In[34]:


np.hstack((arr1, arr2))


# **`np.r_[]`/`np.c_[]` 객체**

# `vstack()`/`hstack()` 과 동일한 기능을 수행하는 특수한 객체들이다.
# 
# 아래 세 개의 어레이를 이용하여 사용법을 살펴본다.

# In[35]:


arr = np.arange(6)
arr


# In[36]:


arr1 = np.arange(6).reshape((3, 2))
arr1


# In[37]:


arr2 = np.random.randn(3, 2)
arr2


# 아래 코드는 `np.vstack((arr1, arr2))`와 동일하다.

# In[38]:


np.r_[arr1, arr2]


# In[39]:


np.vstack([arr1, arr2])


# 아래 코드는 `np.hstack((arr1, arr2))`와 동일하다.

# In[40]:


np.c_[arr1, arr2]


# In[41]:


np.hstack((arr1, arr2))


# 행 또는 열의 크기를 적절하게 맞출 수 있는 어떤 조합도 가능하다.

# In[42]:


np.c_[np.r_[arr1, arr2], arr]


# ## 브로드캐스팅

# 모양이 서로 다른 두 어레이의 연산이 가능한 경우 브로드캐스팅이 작동한다. 
# 예를 들어, 하나의 어레이와 하나의 정수의 곱셈이 항목별로 작동한다. 

# In[43]:


arr = np.arange(6).reshape((2,3))
arr


# 이유는 `arr * 4` 가 아래 어레이의 곱셈과 동일하게 작동하기 때문이다. 
# 즉, 정수 4로 채워진 동일한 모양의 어레이를 먼저 생성한 후에 항목별 곱셈을 진행한다.

# <img src="https://github.com/codingalzi/pydata/blob/master/notebooks/images/broadcasting14.png?raw=true" style="width:300px;">

# In[44]:


arr * 4


# **예제**

# 라애 코드는 1차원 어레이를 2차원 어레이로 확장하여 다른 어레이와 모양을 맞춘 후 연산을 실행하는 것을 보여준다.

# In[45]:


arr2 = np.arange(4).reshape((4,1)).repeat(3,axis=1)
arr2


# In[46]:


arr3 = np.arange(1, 4)
arr3


# In[47]:


arr2 + arr3


# <img src="https://github.com/codingalzi/pydata/blob/master/notebooks/images/broadcasting10.png?raw=true" style="width:400px;">

# In[48]:


arr3a = np.arange(1, 4).reshape((1,3))
arr3a


# In[49]:


arr2 + arr3a


# <img src="https://github.com/codingalzi/pydata/blob/master/notebooks/images/broadcasting10a.png?raw=true" style="width:400px;">

# **예제**

# 아래 예제는 2차원 어레이의 칸을 복제하여 모양을 맞춘 후 연산을 실행한다.

# In[50]:


arr4 = np.arange(1, 5).reshape((4,1))
arr4


# In[51]:


arr2 + arr4


# <img src="https://github.com/codingalzi/pydata/blob/master/notebooks/images/broadcasting11.png?raw=true" style="width:400px;">

# **예제**

# 아래 예제는 2차원 어레이를 3차원으로 확장한 후에 연산을 진행하는 것을 보여준다. 

# In[52]:


arr6 = np.arange(24).reshape((3, 4, 2))
arr6


# In[53]:


arr7 = np.arange(8).reshape((4, 2))
arr7


# In[54]:


arr6 + arr7


# <img src="https://github.com/codingalzi/pydata/blob/master/notebooks/images/broadcasting12.png?raw=true" style="width:400px;">

# **예제**

# In[55]:


arr = np.random.randn(4, 3)
arr


# In[56]:


arr.mean(0)


# In[57]:


demeaned = arr - arr.mean(0)
demeaned


# In[58]:


demeaned.mean(0)


# **예제**

# In[59]:


arr


# In[60]:


row_means = arr.mean(1)
row_means


# In[61]:


row_means.reshape((4, 1))


# In[62]:


demeaned = arr - row_means.reshape((4, 1))


# In[63]:


demeaned.mean(1)


# ### 브로드캐스팅으로 어레이에 값 대입하기

# In[64]:


arr = np.zeros((4, 3))
arr


# In[65]:


arr[:] = 5
arr


# In[66]:


col = np.array([1.28, -0.42, 0.44, 1.6])


# In[67]:


arr[:] = col[:, np.newaxis]
arr


# In[68]:


arr[:2] = [[-1.37], [0.509]]
arr


# ## 연습문제

# 참고: [(실습) 고급 넘파이](https://colab.research.google.com/github/codingalzi/datapy/blob/master/practices/practice-advanced_numpy.ipynb)
