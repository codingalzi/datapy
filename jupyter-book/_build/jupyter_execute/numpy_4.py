#!/usr/bin/env python
# coding: utf-8

# (sec:numpy_4)=
# # 고급 넘파이

# 넘파이 어레이를 조작하는 고급 기능의 일부를 예제를 이용하여 살펴본다.

# **주요 내용**
# 
# * 어레이 변형
# * 차원 추가와 삭제
# * 어레이 이어붙이기/쌓기
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


# In[2]:


# 파이플롯
import matplotlib.pyplot as plt

# 도표 크기 지정
plt.rc('figure', figsize=(10, 6))


# 먼저 어레이를 조작하는 몇 가지 중요한 고급 기법을 살펴본다. 

# ## 어레이 변형

# **`reshape()` 메서드**

# `reshape()` 메서드를 활용하여 주어진 어레이의 모양을 원하는 대로 변형한다.
# 단, 항목의 수가 변하지 않도록 모양을 지정해야 한다.
# 예를 들어, 길이가 8인 1차원 어레이가 다음과 같다.

# In[3]:


arr = np.arange(8)
arr


# 이제 (4, 2) 모양의 2차원 어레이로 모양을 변형할 수 있다.

# In[4]:


arr.reshape((4, 2))


# 항목의 수만 같으면 임의의 차원의 어레이를 임의의 차원의 어레이로 변형시킬 수 있다.

# In[5]:


arr.reshape((4, 2)).reshape((2, 2, 2))


# **`-1`의 역할**

# 어레의 모양을 지정할 때 튜플의 특정 위치에 -1을 사용할 수 있다.
# 그러면 그 위치의 값은 튜플의 다른 항목의 정보를 이용하여 자동 결정된다.
# 예를 들어, 아래 코드에서 -1은 4를 의미한다. 
# 이유는 20개의 항목을 5개의 행으로 이루어진 2차원 어레이로 지정하려면 열은 4개 있어야 하기 때문이다.

# In[6]:


arr = np.arange(20)
arr.reshape((5, -1))


# In[7]:


arr.reshape((5, 4))


# 동일한 이유로 아래에서 -1은 5를 의미한다.

# In[8]:


arr.reshape((2, -1, 2))


# In[9]:


arr.reshape((2, 5, 2))


# ## 차원 추가와 삭제

# 어레이 변형의 특별한 경우로 차원 추가와 삭제가 있다.

# ### 차원 추가

# 어레이에 임의의 축을 추가하는 방식으로 차원이 하나 더 추가된 어레이를 생성할 수 있다.
# 어느 축을 추가하느냐에 따라 생성된 어레이의  모양은 달라진다.

# **예제**

# 다음 길이가 3인 1차원 어레이를 이용하자.

# In[10]:


arr_1d = np.random.normal(size=3)
arr_1d


# `arr_1d`는 원래 0번 축 하나만 갖는데,
# 아래 코드는 여기에 1번 축을 추가하여 2차원 어레이로 만든다.

# In[11]:


arr_1d[:, np.newaxis]


# `reshape()` 메소드로도 동일한 결과를 얻을 수 있다.

# In[12]:


arr_1d.reshape((3, 1))


# 아래 코드는 기존의 0번 축을 1번 축으로 바꾼다.

# In[13]:


arr_1d[np.newaxis, :]


# `reshape()` 메소드로도 동일한 결과를 얻을 수 있다.

# In[14]:


arr_1d.reshape((1,3))


# **예제**

# 2차원 어레이에 축을 추가하면 3차원 어레이가 생성되며, 작동방식은 앞서와 동일하다.

# In[15]:


arr = np.random.normal(size=(4, 3))
arr


# In[16]:


arr[:,:,np.newaxis].shape


# In[17]:


arr[:,:,np.newaxis]


# In[18]:


arr[:,np.newaxis,:].shape


# In[19]:


arr[:,np.newaxis,:]


# ### 차원 삭제

# `ravel()` 메서드와 `flatten()` 메서드는 어레이를 1차원으로 변형한다. 
# 즉, 차원을 모두 없앤다.

# In[20]:


arr = np.arange(15).reshape((5, 3))
arr


# In[21]:


arr1 = arr.ravel()
arr1


# In[22]:


arr2 = arr.flatten()
arr2


# 차이점은 `ravel()` 메서드는 뷰(view)를 사용하는 반면에 `flatten()` 메서드는 어레이를 새로 생성한다.
# 예를 들어, 아래처럼 `arr1`의 항목을 변경하면 `arr`의 항목도 함께 변경된다.

# In[23]:


arr1[0] = -1
arr


# `arr2`은 `arr`과 전혀 상관이 없다.

# In[24]:


arr2[0] = -7
arr


# ## 어레이 이어붙이기와 쌓기

# **`np.concatenate()` 함수**

# 두 개의 어레이를 이어붙인다.
# 지정되는 축에 따라 좌우로 또는 상하로 이어붙인다.
# 아래 세 어레이를 이용하여 사용법을 설명한다.

# In[25]:


arr1 = np.array([[1, 2, 3], [4, 5, 6]])
arr1


# In[26]:


arr2 = np.array([[7, 8, 9], [10, 11, 12]])
arr2


# In[27]:


arr3 = np.array([[13, 14, 15], [16, 17, 18]])
arr3


# 위아래로 이어붙이려면 축을 0으로 정한다.
# 이어붙이 어레이로 이루어진 리스트 또는 튜플을 사용함에 주의한다.

# In[28]:


np.concatenate([arr1, arr2, arr3], axis=0)


# In[29]:


np.concatenate((arr1, arr2, arr3), axis=0)


# 좌우로 이어붙이려면 축을 1로 정한다.

# In[30]:


np.concatenate([arr1, arr2, arr3], axis=1)


# In[31]:


np.concatenate((arr1, arr2, arr3), axis=1)


# **`np.vstack()`/`np.hstack()` 함수**

# 두 함수는 `np.concatenate()` 함수에 축을 각각 0과 1로 지정한 함수이다.

# * `np.vstack((x, y, ...))` := `np.concatenate((x, y, ...), axis=0)`

# In[32]:


np.vstack((arr1, arr2, arr3))


# In[33]:


np.vstack([arr1, arr2, arr3])


# * `np.hstack((x, y, ...))` := `np.concatenate((x, y, ...) axis=1)`

# In[34]:


np.hstack((arr1, arr2, arr3))


# In[35]:


np.hstack([arr1, arr2, arr3])


# **`np.r_[]`/`np.c_[]` 객체**

# `vstack()`/`hstack()` 과 동일한 기능을 수행하는 특수한 객체들이다.

# - `np.r_[x, y, ...]` := `np.vstack((x, y, ...))`

# In[36]:


np.r_[arr1, arr2, arr3]


# - `np.c_[x, y, ...]` := `np.hstack((x, y, ...))`

# In[37]:


np.c_[arr1, arr2, arr3]


# ## 브로드캐스팅

# **브로드캐스팅**<font size='2'>broadcasting</font>
# 모양이 서로 다른 두 어레이가 주어졌을 때 두 모양을 통일시킬 수 있다면 두 어레이의 연산이 가능하도록
# 도와주는 기능이다.
# 설명을 위해 하나의 어레이와 하나의 정수의 곱셈이 작동하는 과정을 살펴본다.

# In[38]:


arr = np.arange(6).reshape((2,3))
arr


# 위 어레이에 4를 곱한 결과는 다음과 같다.

# In[39]:


arr * 4


# 결과가 항목별로 곱해지는 이유는 `arr * 4` 가 아래 어레이의 곱셈과 동일하게 작동하기 때문이다. 
# 즉, 정수 4로 채워진 동일한 모양의 어레이를 먼저 생성한 후에 항목별 곱셈을 진행한다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/datapy/master/jupyter-book/images/broadcasting14.png?raw=true" style="width:300px;"></div>

# 이와 같이 어레이의 모양을 확장하여 항목별 연산이 가능해지도록 하는 기능은 두 어레이의 모야을
# 통일시킬 수 있는 경우 항상 작동한다.

# ### 브로드캐스팅과 연산

# 어레이 연산을 실행할 때 브로드캐스팅이 가능한 경우 자동 적용된다.

# **예제**

# 아래 코드는 1차원 어레이를 2차원 어레이로 확장하여 다른 어레이와 모양을 맞춘 후 연산을 실행한 결과를 보여준다.

# In[40]:


arr2 = np.arange(4).reshape((4,1)).repeat(3,axis=1)
arr2


# In[41]:


arr3 = np.arange(1, 4)
arr3


# In[42]:


arr2 + arr3


# 아래 그림이 위 연산이 작동하는 이유를 설명한다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/datapy/master/jupyter-book/images/broadcasting10.png?raw=true" style="width:400px;"></div>

# 동일한 이유로 다음 연산도 가능하다.

# In[43]:


arr3_a = np.arange(1, 4)[np.newaxis, :]
arr3_a


# In[44]:


arr2 + arr3_a


# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/datapy/master/jupyter-book/images/broadcasting10a.png?raw=true" style="width:400px;"></div>

# **예제**

# 아래 예제는 2차원 어레이의 칸을 복제하여 모양을 맞춘 후 연산을 실행한다.

# In[45]:


arr4 = np.arange(1, 5).reshape((4,1))
arr4


# In[46]:


arr2 + arr4


# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/datapy/master/jupyter-book/images/broadcasting11.png?raw=true" style="width:400px;"></div>

# 반면에 아래 연산은 오류를 발생시킨다.

# In[47]:


x = np.arange(0, 31, 10)
arr5 = np.c_[x, x, x]
arr5


# In[48]:


arr4_a = arr4.flatten()
arr4_a


# ```python
# >>> arr5+ arr4_a
# ---------------------------------------------------------------------------
# ValueError                                Traceback (most recent call last)
# Cell In [80], line 1
# ----> 1 arr5+ arr4_a
# 
# ValueError: operands could not be broadcast together with shapes (4,3) (4,)
# ```

# 아래 그림이 이유를 설명한다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/datapy/master/jupyter-book/images/broadcasting11a.png?raw=true" style="width:380px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://numpy.org/doc/stable/user/basics.broadcasting.html">NumPy: Broadcasting</a>&gt;</div></p>

# 물론 다음은 실행된다.

# In[49]:


arr5 + arr4


# **예제**

# 아래와 같이 두 어레이에 대해 브로디캐스팅을 먼저 적용한 다음에 연산을 실행하기도 한다.

# In[50]:


arr6 = np.arange(0, 31, 10).reshape(4, -1)
arr6


# In[51]:


arr7 = np.arange(1, 4)
arr7


# In[52]:


arr6 + arr7


# 위 연산이 작동하는 이유를 아래 그림이 설명한다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/datapy/master/jupyter-book/images/broadcasting11b.png?raw=true" style="width:500px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://numpy.org/doc/stable/user/basics.broadcasting.html">NumPy: Broadcasting</a>&gt;</div></p>

# **예제**

# 아래 예제는 2차원 어레이를 3차원으로 확장한 후에 연산을 진행하는 것을 보여준다. 

# In[53]:


arr6 = np.arange(24).reshape((3, 4, 2))
arr6


# In[54]:


arr7 = np.arange(8).reshape((4, 2))
arr7


# In[55]:


arr6 + arr7


# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/datapy/master/jupyter-book/images/broadcasting12.png?raw=true" style="width:400px;"></div>

# **예제**

# 아래 코드는 어레이의 열별 평균값이 0이 되도록 하려 한다.

# In[56]:


arr = np.random.randn(4, 3)
arr


# 기존 어레이의 열별 평균값을 각각의 열에서 뺀다.

# In[57]:


arr.mean(0) # arr.mean(axis=0)


# In[58]:


demeaned = arr - arr.mean(0)
demeaned


# 이제 열별 평균값을 확인하면 0이 된다.

# In[59]:


demeaned.mean(0)


# **예제**

# 아래 코드는 어레이의 행별 평균값이 0이 되도록 하려 한다.

# In[60]:


arr


# In[61]:


row_means = arr.mean(1)
row_means


# In[62]:


row_means.reshape((4, 1))


# In[63]:


demeaned = arr - row_means.reshape((4, 1))


# In[64]:


demeaned.mean(1)


# ### 브로드캐스팅과 항목 대체

# 브로드캐스팅으로 어레이의 항목을 대체할 수 있다.
# 설명을 위해 아래 어레이를 사용한다.

# In[65]:


arr = np.zeros((4, 3))
arr


# **예제**

# 모든 항목을 5로 대체한다.

# In[66]:


arr[:] = 5
arr


# **예제**

# 모든 열을 지정된 열로 대체한다.

# In[67]:


col = np.array([1.28, -0.42, 0.44, 1.6])
col[:, np.newaxis]


# In[68]:


arr[:] = col[:, np.newaxis]
arr


# **예제**

# 0번, 1번 행을 특정 값으로 대체한다.

# In[69]:


arr[:2] = [[-1.37], [0.509]]
arr


# ## 연습문제

# 참고: [(실습) 고급 넘파이](https://colab.research.google.com/github/codingalzi/datapy/blob/master/practices/practice-numpy_4.ipynb)
