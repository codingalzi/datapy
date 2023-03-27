#!/usr/bin/env python
# coding: utf-8

# (sec:numpy_1)=
# # 넘파이 어레이

# **주요 내용**
# 
# - 넘파이 어레이 소개
# - 어레이 기초 연산

# **슬라이드**
# 
# 본문 내용을 요약한 [슬라이드](https://github.com/codingalzi/datapy/raw/master/slides/slides-numpy_1.pdf)를 다운로드할 수 있다.

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


# ## 넘파이란?

# **넘파이**<font size='2'>numpy</font>는 NUMerical PYthon의 줄임말이며, 
# 파이썬 데이터 과학에서 가장 중요한 도구를 제공하는 라이브러리이다.
# 넘파이가 제공하는 가장 중요한 요소는 아래 두 가지이다.
# 
# * 다차원 어레이(배열)
# * 메모리 효율적인 빠른 어레이 연산
# 
# 넘파이의 기능을 잘 이해한다면 이어서 다룰 **판다스**<font size='2'>pandas</font> 라이브러리가 지원하는
# 데이터프레임<font size='2'>Dataframe</font> 자료형의 기능을 매우 쉽게 활용할 수 있다. 

# 리스트 연산과 넘파이 어레이 연산의 속도 차이를 아래 코드가 보여준다.
# 아래 코드는 0부터 999,999까지의 숫자를 각각 두 배하는 연산에 필요한 시간을 측정한다.
# 결과적으로 넘파이 어레이를 이용한 연산이 50배 정도 빠르다.

# In[3]:


my_array = np.arange(1000000)
my_list = list(range(1000000))


# :::{admonition} `%time` 매직 커맨드
# :class: warning
# 
# `%time`은 코드 실행시간을 측정하는 IPython의 매직 커맨드 중의 하나이며, 파이썬 자체의 기능이 아니다.
# :::

# 아래 코드에서 `my_array * 2`는 `my_array` 어레이의 항목 각각을 두 배한 값을 항목으로 갖는다.

# In[4]:


get_ipython().run_line_magic('time', 'for _ in range(10): my_array2 = my_array * 2')


# In[5]:


get_ipython().run_line_magic('time', 'for _ in range(10): my_list2 = [x * 2 for x in my_list]')


# 이제부터 넘파이에 대해 필수적으로 알아 두어야만 하는 내용들을 정리하며 살펴본다.

# ## 다차원 어레이

# 리스트, 튜플 등을 `np.array()` 함수를 이용하여 어레이로 변환시킬 수 있다.

# ### 1차원 어레이

# * 리스트 활용

# In[6]:


data1 = [6, 7.5, 8, 0, 1]
arr1 = np.array(data1)
arr1


# * 튜플 활용

# In[7]:


data1 = (6, 7.5, 8, 0, 1)
arr1 = np.array(data1)
arr1


# **`ndarray` 자료형**
# 
# 넘파이 어레이의 자료형은 `ndarray`이다. 

# In[8]:


type(arr1)


# ### 2차원 어레이

# 중첩된 리스트를 2차원 어레이로 변환할 수 있다.
# 단, 항목으로 사용된 리스트의 길이가 모두 동일해야 한다.
# 즉, 2차원 어레이는 어레이의 모든 항목이 동일한 크기의 1차원 어레이이다.

# In[9]:


data2 = [[1, 2, 3, 4], [5, 6, 7, 8]]
arr2 = np.array(data2)
arr2


# **`shape`** 속성
# 
# 어레이 객체의 `shape` 속성은 생성된 어레이의 모양을 저장한다.
# 2차원 어레이는 행과 열의 크기를 이용한 튜플로 보여준다.
# 위 어레이의 모양(shape)은 (2, 4)이다.
# 
# * 2: 항목이 두개
# * 4: 각각의 항목은 길이가 4인 어레이

# In[10]:


arr2.shape


# 1차원 어레이의 모양은 어레이의 길이 정보를 길이가 1인 튜플로 저장한다.

# In[11]:


arr1.shape


# **`dtype`** 속성
# 
# 어레이 객체의 `dtype` 속성은 어레이에 사용된 항목들의 자료형을 저장한다.
# 어레이의 모든 항목은 **동일한 자료형**을 가져야 한다.
# 넘파이는 파이썬 표준에서 제공하는 자료형보다 세분화된 자료형을 지원한다. 
# 예를 들어, `float64`는 64비트로 구현된 부동소수점 자료형을 가리킨다.
# 앞으로 계속해서 세분화된 자료형을 만나게 될 것이다.

# `arr2` 항목의 자료형은 `int64`, 즉, 64비트로 구현된 정수 자료형이다.

# In[12]:


arr2.dtype


# **`ndim` 속성**

# 차원은 `ndim` 속성에 저장되며, `shape`에 저정된 튜플의 길이와 동일하다.

# In[13]:


arr2.ndim


# 1차원 어레이의 차원은 1이다.

# In[14]:


arr1.ndim


# ### 3차원 어레이

# 3차원 이상의 어레이는 처음에는 매우 생소하게 다가올 수 있다.
# 하지만 이미지 데이터 분석에서 가장 기본으로 사용되는 차원이기에 3차원 어레이에 익숙해져야 한다.
# 
# (n, m, p) 모양의 3차원 어레이를 이해하는 두 가지 방법은 다음과 같다.
# 
# 첫째, 바둑판을 (n, m) 크기의 격자로 나누고 각각의 칸에 길이가 p 인 1차원 어레이가 
# 위치하는 것으로 이해한다. 
# 아래 그림은 (3, 3, 3) 모양의 어레이이며, (3, 3) 모양의 바둑판에 
# 0과 1사이의 부동소수점으로 구성된 (R, G, B) 색상 정보를 담은 길이가 3인 튜플을 표현한다.

# <div align="center" border="1px"><img src="https://raw.githubusercontent.com/codingalzi/datapy/master/jupyter-book/images/impixelregion.png" style="width:5=700px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="http://maprabu.blogspot.com/2013/08/dont-photoshop-just-matlab-it.html">Big Data & Image Processing</a>&gt;</div></p>

# 둘째, (m, p) 모양의 2차원 어레이 n 개를 항목으로 갖는 1차원 어레이로 이해한다.
# 아래 `threeD_array` 변수가 가리키는 3차원 어레이는 (3, 2) 모양의 2차원 어레이를 4개 포함한 1차원 어레이로 이해할 수 있다.

# <div align="center" border="1px"><img src="https://raw.githubusercontent.com/codingalzi/datapy/master/jupyter-book/images/numpy-array.svg" style="width:30%;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.reallifeedublogging.com/2020/07/numpy-arrays-and-data-analysis.html">NumPy Arrays and Data Analysis</a>&gt;</div></p>

# In[15]:


threeD_array = np.array([[[7, 1],
                          [9, 4], 
                          [2, 3]],
   
                         [[4, 3],
                          [0, 1], 
                          [8, 0]],

                         [[1, 2], 
                          [3, 4], 
                          [6, 9]],

                         [[5, 11], 
                          [8, 5], 
                          [4, 7]]])


# In[16]:


threeD_array.shape


# :::{admonition} 어레이의 차원 대 벡터의 차원
# :class: info
# 
# 벡터는 1차원 어레이에 해당한다.
# 그리고 벡터의 차원은 벡터에 포함된 항목의 개수, 
# 즉 1차원 어레이의 모양<font size='2'>shape</font>에 사용되는 항목의 값이다.
# 따라서 벡터의 차원과 어레이의 차원은 기본적으로 아무 상관이 없다.
# :::

# ### 어레이 객체 생성 함수

# 배열을 쉽게 생성할 수 있는 함수는 다음과 같으며, 
# 각 함수의 기능은 
# [numpy cheat sheet](https://ipgp.github.io/scientific_python_cheat_sheet/?utm_content=buffer7d821&utm_medium=social&utm_source=twitter.com&utm_campaign=buffer#numpy-import-numpy-as-np)를 
# 참고한다.
# 
# * `np.array()`
# * `np.arange()`
# * `np.diag()`
# * `np.empty()`
# * `np.ones()`
# * `np.zeros()`

# `np.array()` 함수는 앞서 소개하였으며
# 추가로 `np.zeros()`, `np.arange()` 두 함수를 간략하게 다룬다.
# 나머지 함수들은 연습문제를 통해 학습하도록 한다.

# **`zeros()` 함수**
# 
# 0으로 이루어진 어레이를 생성한다. 
# 1차원인 경우 정수를 인자로 사용한다.

# In[17]:


np.zeros(10)


# 2차원부터는 정수들의 튜플로 모양을 지정한다.

# In[18]:


np.zeros((3, 6))


# In[19]:


np.zeros((4, 3, 2))


# **`arange()` 함수**
# 
# `range()` 함수와 유사하게 작동하며 부동소수점 스텝도 지원한다.

# In[20]:


np.arange(15)


# In[21]:


np.arange(0, 1, 0.1)


# ## `dtype` 종류

# `dtype` 속성은 어레이 항목의 자료형을 담고 있으며, 파이썬 표준 라이브러리에서 제공하는 
# `int`, `float`, `str`, `bool` 등을 보다 세분화시킨 자료형을 제공한다.

# ### 기본 `dtype`

# 여기서는 세분화된 자료형을 일일이 설명하기 보다는 예제를 이용하여 세분화된 자료형의 형식을 
# 살펴본다.
# 자료형 세분화는 주로 자료형의 객체가 사용하는 메모리 용량을 제한하는 형식으로 이루어진다. 
# 이를 통해 보다 메모리 효율적이며 빠른 계산이 가능해졌다.

# | 자료형 | 자료형 코드 | 설명 |
# | :--- | :--- | :--- |
# | int8 / uint8 | 'i1' / 'u1' | signed / unsigned 8 비트 정수|
# | int16 / uint16 | 'i2' / 'u2' | signed / unsigned 16 비트 정수|
# | int32 / uint32 | 'i4' / 'u4' | signed / unsigned 32 비트 정수|
# | int64 / uint64 | 'i8' / 'u8' | signed / unsigned 64 비트 정수|
# | float16 | 'f2' | 16비트(반 정밀도) 부동소수점 |
# | float32 | 'f4' 또는 'f' | 32비트(단 정밀도) 부동소수점 |
# | float64 | 'f8' 또는 'd' | 64비트(배 정밀도) 부동소수점 |
# | float128 | 'f16' 또는 'g' | 64비트(배 정밀도) 부동소수점 |
# | bool | '?' | 부울 값 |
# | object | 'O' | 임의의 파이썬 객체 |
# | string_ | 'S' | 고정 길이 아스키 문자열 / 예) `S8`, `S10` |
# | unicode_ | 'U' | 고정 길이 유니코드 문자열 / 예) `U8`, `U10`|

# **`float64` 자료형**

# In[22]:


arr1 = np.array([1, 2, 3], dtype=np.float64)

arr1.dtype


# In[23]:


arr1 = np.array([1, 2, 3], dtype='f8')

arr1.dtype


# **`int32` 자료형**

# In[24]:


arr2 = np.array([1, 2, 3], dtype=np.int32)

arr2.dtype


# In[25]:


arr2 = np.array([1, 2, 3], dtype='i4')

arr2.dtype


# **문자열 자료형**

# 문자열은 기본적으로 유니코드로 처리되며 크기는 최장 길이의 문자열에 맞춰 결정된다.

# In[26]:


np.array(['python', 'data']).dtype


# In[27]:


numeric_strings = np.array(['1.25', '-9.6', '42'])
numeric_strings.dtype


# **`bool` 자료형**

# In[28]:


np.array([True, False], dtype='?').dtype


# ### 형변환: `astype()` 메서드

# `astype()` 메서드를 이용하여 dtype을 변경할 수 있다.
# 즉, 항목의 자료형을 강제로 변환시킨다.
# 
# * `int` 자료형을 `float` 자료형으로 형변환하기

# In[29]:


arr = np.array([1, 2, 3, 4, 5])
arr.dtype


# In[30]:


float_arr = arr.astype(np.float64)
float_arr.dtype


# * `float` 자료형을 `int` 자료형으로 형변환하기
#     - 소수점 이하는 버림.

# In[31]:


arr = np.array([3.7, -1.2, -2.6, 0.5, 12.9, 10.1])
arr


# In[32]:


arr.astype(np.int32)


# * 숫자 형식의 문자열을 숫자로 형변환하기: 문자열 자료형의 크기는 넘파이가 알아서 정함

# In[33]:


numeric_strings = np.array(['1.25', '-9.6', '42'])
numeric_strings.dtype


# In[34]:


numeric_strings.astype(float)


# In[35]:


numeric_strings2 = np.array(['1.25345', '-9.673811345', '42'], dtype=np.string_)
numeric_strings2.dtype


# * 부동소수점으로 형변환하면 
#     지정된 정밀도에 따라 소수점 이하를 자른다.

# In[36]:


numeric_strings2.astype(float)


# **주의사항:** 앞서 부동소수점 정밀도를 4로 지정했기 때문에 어레이 항목은 모두 소수점 이하 네 자리까지만 보여준다.
# 
# ```python
# np.set_printoptions(precision=4, suppress=True)
# ```
# 
# 부동소수점 정밀도를 변경하면 그에 따라 다르게 결정된다.

# In[37]:


np.set_printoptions(precision=6, suppress=True)


# In[38]:


numeric_strings2.astype(float)


# `astype()` 메서드의 인자로 다른 배열의 `dtype` 정보를 이용할 수도 있다.

# In[39]:


int_array = np.arange(10)
int_array.dtype


# In[40]:


calibers = np.array([.22, .270, .357, .380, .44, .50], dtype=np.float64)


# In[41]:


int_array.astype(calibers.dtype)


# 자료형 코드를 이용하여 `dtype`을 지정할 수 있다. (위 테이블 참조)

# In[42]:


empty_uint32 = np.empty(8, dtype='u4')
empty_uint32.dtype


# ## 어레이 연산

# 넘파이 어레이 연산은 기본적으로 항목별로 이루어진다. 
# 즉, 지정된 연산을 동일한 위치의 항목끼리 실행하여 새로운, 동일한 모양의 어레이를 생성한다.

# In[43]:


arr = np.array([[1., 2., 3.], [4., 5., 6.]])
arr


# In[44]:


arr2 = np.array([[3., 2., 1.], [4., 2., 12.]])
arr2


# **덧셈**

# In[45]:


arr + arr2


# 숫자와의 연산은 모든 항목에 동일한 값을 사용한다.

# In[46]:


arr + 2.4


# **뺄셈**

# In[47]:


arr - arr2


# In[48]:


3.78 - arr


# **나눗셈**

# 나눗셈 또한 항목별로 연산이 이루어진다. 
# 따라서 0이 항목으로 포함되면 오류가 발생한다.

# In[49]:


arr / arr2


# In[50]:


1 / arr


# In[51]:


arr / 3.2


# **거듭제곱(지수승)**

# In[52]:


arr ** arr2


# In[53]:


2 ** arr


# In[54]:


arr ** 0.5


# **비교 연산**

# In[55]:


arr2 > arr


# In[56]:


arr2 <= arr


# In[57]:


1.2 < arr


# In[58]:


1.2 >= arr2


# In[59]:


arr == arr


# In[60]:


arr != arr2


# **논리 연산**

# 사용 가능한 논리 연산은 아래 세 가지이다.
# 
# * `~`: 부정(not) 연산자
# * `&`: 논리곱(and) 연산자
# * `|`: 논리합(or) 연산자

# In[61]:


~(arr == arr)


# In[62]:


(arr == arr) & (arr2 == arr2)


# In[63]:


~(arr == arr) | (arr2 != arr)


# ## 연습문제

# 참고: [(실습) 넘파이 어레이](https://colab.research.google.com/github/codingalzi/datapy/blob/master/practices/practice-numpy_1.ipynb)
