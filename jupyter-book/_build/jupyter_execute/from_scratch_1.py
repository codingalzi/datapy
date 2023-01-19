#!/usr/bin/env python
# coding: utf-8

# (sec:from_scratch_1)=
# # 선형대수 기초

# **참고** 
# 
# 여기서 사용하는 코드는 조엘 그루스(Joel Grus)의 
# [밑다닥부터 시작하는 데이터 과학](https://github.com/joelgrus/data-science-from-scratch) 
# 4장에 사용된 소스코드의 일부를 기반으로 작성되었다.

# **주요 내용**
# 
# 넘파이 어레이(`numpy.array`)는 길이와 모양에 대한 정보와 함께 어레이를 조작하거나 
# 어레이로부터 다양한 정보를 추출하는 메서드를 기본으로 제공한다.
# 반면에 파이썬 리스트(`list`)는 인덱싱과 몇 개의 리스트 조작 기능 이외에 
# 별 다른 정보와 기능을 제공하지 않는다. 
# 
# 여기서는 벡터와 행렬을 각각 1차원과 2차원 리스트로 구현하여 실용적으로 사용하는 과정을 살펴본다.
# 보다 구체적으로는 벡터와 행렬의 자료형 정의에서 출발하여 벡터와 행렬의 연산 등을 모두 
# 리스트와 기본 파이썬만을 이용하여 구현한다. 
# 
# 이를 통해 넘파이 어레이가 제공하는 다양한 기능에 대한 보다 깊은 이해와 함께 파이썬 데이터 분석 관련
# 프로그래밍 실력 향상에 도움을 주고자 한다.

# **주의사항** 
# 
# 아래 코드는 
# [리스트 조건제시법](https://codingalzi.github.io/pybook/collections2.html#sec-comprehension)을 
# 활용한다. 

# ## 벡터

# 벡터는 유한 개의 값으로 구성된다.
# 보통 수를 항목으로 사용하며, 항목의 수를 벡터의 **차원**<font size='2'>dimension</font>이라 부른다.
# 벡터는 수학, 통계, 물리 등 과학 분야에서 많이 사용되며, 
# 최근에 컴퓨터 데이터 분석이 발전하면서 벡터의 활용도 매우 높아졌다.
# 
# * 2차원 평면 공간에서 방향과 크기를 표현하는 2차원 벡터: (x, y)
# * 사람들의 키, 몸무게, 나이로 이루어진 3차원 벡터: (키, 몸무게, 나이)
# * 네 번의 시험 점수로 이루어진 4차원 벡터: (1차점수, 2차점수, 3차점수, 4차점수)

# ### 리스트와 벡터

# 리스트를 이용하여 벡터를 구현할 수 있다.

# **예제**
# 
# x축, y축 좌표로 구성된 2차원 벡터

# In[1]:


# (x축, y축)

twoDVector1 = [3, 1]
twoDVector2 = [-2, 5]


# **예제**
# 
# 키, 몸무게, 나이로 구성된 3차원 벡터

# In[2]:


# (키, 몸무게, 나이)

height_weight_age1 = [70, 170, 50]
height_weight_age2 = [66, 163, 50]


# **예제**
# 
# 1차부터 4차까지의 시험 점수로 구성된 4차원 벡터

# In[3]:


# (1차점수, 2차점수, 3차점수, 4차점수)

grades1 = [95, 80, 75, 62]
grades2 = [85, 82, 79, 82]


# ### 벡터 연산

# **벡터 덧셈**

# 차원이 같은 벡터 두 개를 더할 수 있다.
# 두 벡터의 덧셈은 같은 위치에 있는 항목끼기 더한 결과로 이루어진 벡터를 생성한다.
# 벡터 $u = (u_1, \cdots, u_n)$와 벡터 $v = (v_1, \cdots, v_n)$가 주어졌을 때
# 두 벡터의 덧셈은 아래와 같다. 
# 
# $$
# u + v 
# = (u_1 + v_1, \cdots, u_n + v_n)
# $$

# **벡터 덧셈 함수**
# 
# 차원이 같은 두 벡터의 항목별 합을 항목으로 같은 벡터를 반환하는 함수는 다음과 같다.

# In[4]:


def addV(v, w):
    assert len(v) == len(w)   # 두 벡터의 길이가 같은 경우만 취급

    return [v_i + w_i for v_i, w_i in zip(v, w)]


# In[5]:


addV(twoDVector1, twoDVector2)


# In[6]:


addV(height_weight_age1, height_weight_age2)


# In[7]:


addV(grades1, grades2)


# **벡터 리스트의 합**

# 동일한 차원의 임의의 개수의 벡터를 더하는 함수를 다음과 같이 정의할 수 있다.
# 함수의 본문에 사용된 `all()` 함수는 리스트, 튜플 등에 포함된 모든 항목이 참인 경우에만
# 참을 반환한다.

# In[8]:


def vector_sum(vectors):
    """
    vectors: 동일한 차원의 벡터들의 리스트
    반환값: 각 항목의 합으로 이루어진 동일한 차원의 벡터
    """
    
    # 입력값 확인
    assert len(vectors) > 0          # 1개 이상의 벡터가 주어져야 함
    num_elements = len(vectors[0])   # 벡터 개수
    assert all(len(v) == num_elements for v in vectors)   # 모든 벡터의 크기가 같아야 함

    # 동일한 위치의 항목을 모두 더한 값들로 이루어진 벡터 반환
    return [sum(vector[i] for vector in vectors) for i in range(num_elements)]


# **예제: 네 개의 2차원 벡터 덧셈**
# 
# 예를 들어, 2차원 벡터 네 개를 더한 결과는 다음과 같다.

# In[9]:


vector_sum([[1, 2], [3, 4], [5, 6], [7, 8]])


# **벡터 뺄셈**

# 차원이 같은 벡터에 대해 뺄셈을 정의할 수 있다.
# 차원이 같은 벡터 두 개의 덧셈은 같은 위치에 있는 항목끼기 뺀 결과로 이루어진 벡터를 생성한다.
# 벡터 $u = (u_1, \cdots, u_n)$와 벡터 $v = (v_1, \cdots, v_n)$가 주어졌을 때
# 두 벡터의 뺄셈은 아래와 같다. 
# 
# $$
# u - v 
# = (u_1 - v_1, \cdots, u_n - v_n)
# $$

# **벡터 뺄셈 함수**
# 
# 차원이 같은 두 벡터의 항목별 차를 항목으로 같은 벡터를 반환하는 함수는 다음과 같다.

# In[10]:


def subtractV(v, w):
    assert len(v) == len(w)   # 두 벡터의 길이가 같은 경우만 취급

    return [v_i - w_i for v_i, w_i in zip(v, w)]


# In[11]:


subtractV(twoDVector1, twoDVector2)


# In[12]:


subtractV(height_weight_age1, height_weight_age2)


# In[13]:


subtractV(grades1, grades2)


# **벡터 스칼라 곱셈**

# 하나 수와 하나의 벡터의 곱셈을 스칼라 곱셈이라 부른다. 
# 스칼라 곱셈은 벡터의 각 항목을 지정된 수로 곱한다.
# 벡터 $u = (u_1, \cdots, u_n)$와 하나의 수 $c$가 주어졌을 때
# 벡터 스칼라 곱셈은 아래와 같다. 
# 
# $$
# c\cdot u
# = (c\cdot u_1, \cdots, c\cdot u_n)
# $$

# **벡터 스칼라 곱셈 함수**
# 
# 벡터의 각 항목에 동일한 부동소수점을 곱한 결과를 반화하는 함수는 다음과 같다.

# In[14]:


def scalar_multV(c, v):
    return [c * v_i for v_i in v]


# In[15]:


scalar_multV(2, [1, 2, 3])


# **항목별 평균 벡터**

# 동일한 차원의 벡터가 여러 개 주어졌을 때 항목별 평균을 구할 수 있다.
# 항목별 평균은 항목끼리 모두 더한 후 벡터의 개수로 나눈다.

# **예제:3개의 2차원 벡터들의 평균**

# $$
# \frac 1 3 \cdot \left ((1, 2) + (2, 1) + (2, 3) \right) 
# =  \frac 1 3 \cdot (1+2+2, 2+1+3) 
# = \left (\frac 5 3, 2 \right)
# $$

# **항목별 평균 벡터 함수**
# 
# 동일한 차원의 여러 벡터가 주어졌을 때 항목별 평균으로 이루어진 벡터를 반환하는 함수는 다음과 같이
# 벡터들의 합을 벡터의 개수로 나눈다.

# In[16]:


def vector_mean(vectors):
    n = len(vectors)
    
    return scalar_multV(1/n, vector_sum(vectors))


# In[17]:


vector_mean([[3, 2], [2, 5], [7, 5], [6, 3]])


# ### 벡터 내적과 크기

# 차원이 같은 벡터 두 개의 내적은 같은 위치에 있는 항목끼기 곱한 후 모두 더한 값이다.
# 벡터 $u = (u_1, \cdots, u_n)$와 벡터 $v = (v_1, \cdots, v_n)$가 주어졌을 때
# 두 벡터의 내적은 아래와 같다. 
# 
# $$
# u \cdot v 
# = \sum_{i=1}^n u_i\cdot v_i 
# = u_1\cdot v_1 + \cdots + u_n\cdot v_n
# $$

# **벡터 내적 함수**
# 
# 동일 차원의 두 벡터의 내적을 반환하는 함수는 다음과 같다.

# In[18]:


def dot(v, w):
    assert len(v) == len(w), "벡터들의 길이가 동일해야 함"""

    return sum(v_i * w_i for v_i, w_i in zip(v, w))


# In[19]:


dot([1, 2, 3], [4, 5, 6])


# **벡터의 크기**

# 벡터 $v = (v_1, \cdots, v_n)$가 주어졌을 때 
# 벡터 $v$의 크기 $\| v\|$는 $v$ 자신과의 내적의 제곱근이다.
# 
# $$\| v\| = \sqrt{v \cdot v} = \sqrt{v_1^2 + \cdots + v_n^2}$$

# **예제**
# 
# 벡터 $(3, 4)$의 크기는 다음과 같다.
# 
# $$\| (3, 4)\|  = \sqrt{3^2 + 4^2} = \sqrt{5^2} = 5$$

# **벡터의 크기 계산 함수**
# 
# 제곱근은 `math` 모듈의 `sqrt()` 함수를 이용한다.
# 참고로 벡터의 크기는 영어로 magnitude로 표기한다.

# In[20]:


import math

def magnitude(v):
    sum_of_squares = dot(v, v)
    return math.sqrt(sum_of_squares)


# In[21]:


magnitude([3, 4])


# ## 행렬

# **행렬**<font size='2'>matrix</font>은 숫자를 행과 열로 구성된 직사각형 모양으로 나열한 것이다. 
# $n$ 개의 행과 $k$ 개의 열로 구성된 행렬을 $n \times k$ 행렬이라 부른다.
# 대부분의 프로그래밍 언어에서 행렬을 리스트의 리스트, 즉 2중 리스트로 구현한다.
# 예를 들어 아래 코드에서 `A`는 $2 \times 3$ 행렬이고, `B`는 $3 \times 2$ 행렬이다.

# In[22]:


# 2x3 행렬

A = [[1, 2, 3],
     [4, 5, 6]]


# In[23]:


# 3x2 행렬

B = [[1, 2],
     [3, 4],
     [5, 6]]


# **행렬의 모양**

# $n \times k$ 행렬의 **모양**<font size='2'>shape</font>을 $(n,k)$로 표기한다.
# 예를 들어, $1, 2, 3, 4, 5, 6$ 여섯 개의 항목을 가진 행렬의 모양은 네 종류인데, 
# 이유는 6을 두 개의 양의 정수의 곱셈으로 표현하는 방법이 네 가지이기 때문이다. 

# * (1, 6) 모양의 행렬: 한 개의 행과 여섯 개의 열
# 
# $$
# \begin{bmatrix}
#     1 & 2 & 3 & 4 & 5 & 6
# \end{bmatrix}
# $$

# * (2, 3) 모양의 행렬: 두 개의 행과 세 개의 열
# 
# $$
# \begin{bmatrix}
#     1 & 2 & 3\\
#     4 & 5 & 6
# \end{bmatrix}
# $$

# * (3, 2) 모양의 행렬: 세 개의 행과 두 개의 열
# 
# $$
# \begin{bmatrix}
#     1 & 2 \\
#     3 & 4 \\
#     5 & 6
# \end{bmatrix}
# $$

# * (6, 1) 모양의 행렬: 여섯 개의 행과 한 개의 열
# 
# $$
# \begin{bmatrix}
#     1 \\
#     2 \\
#     3 \\
#     4 \\
#     5 \\
#     6
# \end{bmatrix}
# $$

# **행렬의 모양 확인 함수**

# 아래 코드의 `shape()` 함수는 주어진 행렬의 모양을 튜플로 반환한다.

# In[24]:


def shape(M):
    """
    M: 행렬
    M[i]의 길이가 일정하다고 가정
    """

    num_rows = len(M)    # 행의 수
    num_cols = len(M[0]) # 열의 수
    return num_rows, num_cols


# In[25]:


shape(A)


# In[26]:


shape(B)


# **행벡터와 열벡터**
# 
# 아래 두 함수는 각각 지정된 행와 지정된 열의 항목들로 구성된 행벡터와 열벡터를 반환한다.

# In[27]:


# i번 행벡터
def get_row(M, i):
    """
    M: 행렬
    i: 행 인덱스
    """

    return M[i]             

# j번 열벡터
def get_column(M, j):
    """
    M: 행렬
    j: 열 인덱스
    """

    return [M_i[j] for M_i in M]


# 행렬 `A`의 0번 행은 다음과 같다.

# In[28]:


get_row(A, 0)


# 행렬 `B`의 1번 열은 다음과 같다.

# In[29]:


get_column(B, 1)


# ### 행렬 초기화

# 경우에 따라 0으로만, 1로만, 또는 무작위 수로 구성된 특정 모양의 행렬을 필요하다.
# `make_matrix()` 함수는 행렬의 항목을 생성하는 방식을 지정하면
# 원하는 모양의 행렬을 생성한다.
# 
# * 인자: 3개의 인자가 사용된다.
#     * `n`: 행의 수
#     * `m`: 열의 수
#     * `entry_fn`: i, j가 주어지면 i행, j열에 위치한 값 계산
# * 반환값: 지정된 방식으로 계산된 (i, j) 모양의 행렬

# In[30]:


def make_matrix(n, m, entry_fn):
    """
    n: 행의 수
    m: 열의 수
    entry_fn: (i, j)에 대해 i행, j열에 위치한 항목 계산
    """
    
    return [ [entry_fn(i, j) for j in range(m)] for i in range(n) ]   


# **0-행렬**
# 
# 0-행렬<font size='2'>zero matrix</font>이란 행렬의 모든 원소의 값이 0인 행렬을 말한다.
# 예를 들어 아래 행렬은 (3, 2) 모양의 0-행렬이다.
# 
# $$
# \begin{bmatrix}
#     0 & 0 \\
#     0 & 0 \\
#     0 & 0
# \end{bmatrix}
# $$

# 지정된 모양의 0-행렬을 생성하는 함수는 다음과 같다.

# In[31]:


def zeros(x):
    """
    x = (n, m), 단 n, m은 양의 정수
    """

    n = x[0]
    m = x[1]
    
    zero_function = lambda i, j: 0
    
    return make_matrix(n, m, zero_function)


# In[32]:


zeros((5,7))


# **1-행렬**
# 
# 1-행렬<font size='2'>one matrix</font>이란 행렬의 모든 원소의 값이 1인 행렬을 말한다.
# 예를 들어 아래 행렬은 (3, 4) 모양의 1-행렬이다.
# 
# $$
# \begin{bmatrix}
#     1 & 1 & 1 & 1 \\
#     1 & 1 & 1 & 1 \\
#     1 & 1 & 1 & 1
# \end{bmatrix}
# $$

# 지정된 모양의 1-행렬을 생성하는 함수는 다음과 같다.

# In[33]:


def ones(x):
    """
    x = (n, m), 단 n, m은 양의 정수
    """

    n = x[0]
    m = x[1]
    
    one_function = lambda i, j: 1
    
    return make_matrix(n, m, one_function)


# In[34]:


ones((5,7))


# **임의행렬**

# 임의행렬<font size='2'>random matrix</font>은 행렬의 항목이 임의의 수로 구성된 행렬을 가리킨다.
# 여기서는 0과 1 사이의 임의의 수로만 구성된 임의행렬을 생성한다.
# 이를 위해 `random` 모듈의 `random()` 함수를 이용한다.

# In[35]:


import random


# `random.random()` 함수는 [0, 1) 구간에서 임의의 수를 무작위로 생성한다.

# In[36]:


random.random()


# In[37]:


random.random()


# 지정된 모양의 임의행렬을 생성하는 함수는 다음과 같다.

# In[38]:


def rand(n, m):
    """
    n, m: 양의 정수
    """

    random_function = lambda i, j: random.random()

    return make_matrix(n, m, random_function)


# In[39]:


rand(5,3)


# 부동소수점을 소수점 아래 몇 자리까지만 보이도록 하기 위해 `round()` 함수를 이용할 수 있다.
# `rand()` 함수를 재정의 한다. 
# 이때 항목을 생성하는 함수에 필요한 소수점 이하 자릿수를 `ndigits=2` 키워드 매개변수가
# 받도록 한다. 기본값은 2로 지정한다.

# In[40]:


def rand(n, m, ndigits=2):
    """
    n, m: 양의 정수
    """

    random_function = lambda i, j: round(random.random(), ndigits)  # ndigits: 소수점 이하 자릿수

    return make_matrix(n, m, random_function)


# 소수점 이하 셋째 자리에서 반올림한 값을 사용하도록 하려면
# 기본 인자 2를 그대로 사용하면 되기에 굳이 셋째 인자를 지정할 필요가 없다.

# In[41]:


rand(5, 3)


# 소수점 아래 다섯째 자리까지 보이도록 하려면 셋째 인자를 5로 지정한다.

# In[42]:


rand(5, 3, 5)


# 키워드 매개변수 이름을 함께 사용해도 된다.

# In[43]:


rand(5, 3, ndigits=5)


# **항등행렬**
# 
# 항등행렬<font size='2'>identity matrix</font>은 정사각형 모양의 행렬 중에서 대각선 상에 위치한 항목은 1이고
# 나머지는 0인 행렬을 말한다. 
# 예를 들어 아래 행렬은 (5, 5) 모양의 단위행렬이다.
# 
# $$
# \begin{bmatrix}
#     1&0&0&0&0 \\
#     0&1&0&0&0 \\
#     0&0&1&0&0 \\
#     0&0&0&1&0 \\
#     0&0&0&0&1
# \end{bmatrix}
# $$
# 
# 단위행렬은 행과 열의 개수가 동일한 정방행렬이며,
# 지정된 모양의 단위행렬을 생성하는 함수는 다음과 같다.

# In[44]:


def identity(n):
    """
    n: 양의 정수
    """
    one_function = lambda i, j: 1 if i == j else 0
    
    return make_matrix(n, n, one_function)


# In[45]:


identity(5)


# ### 행렬 연산

# **행렬의 덧셈과 뺄셈**
# 
# 모양이 같은 두 행렬의 덧셈/뺄셈은 항목별로 더한/뺀 결과로 이루어진 행렬이다.
# 즉, 벡터의 덧셈/뺄셈과 동일한 방식이다.
# 예를 들어, $2 \times 3$ 행렬의 덧셈/뺄셈은 다음과 같다.

# * 행렬 덧셈
# 
#     $$
#     \begin{align*}
#     \begin{bmatrix}1&3&7\\1&0&0\end{bmatrix} 
#     + \begin{bmatrix}0&0&5\\7&5&0\end{bmatrix}
#     &= \begin{bmatrix}1+0&3+0&7+5\\1+7&0+5&0+0\end{bmatrix} \\[.5ex]
#     &= \begin{bmatrix}1&3&12\\8&5&0\end{bmatrix}
#     \end{align*}
#     $$

# * 행렬 뺄셈
# 
#     $$
#     \begin{align*}
#     \begin{bmatrix}1&3&7\\1&0&0\end{bmatrix} 
#     - \begin{bmatrix}0&0&5\\7&5&0\end{bmatrix}
#     &= \begin{bmatrix}1-0&3-0&7-5\\1-7&0-5&0-0\end{bmatrix} \\[.5ex]
#     &= \begin{bmatrix}1&3&2\\-6&-5&0\end{bmatrix}
#     \end{align*}
#     $$

# 행렬의 덧셈과 뺄셈을 계산하는 함수는 다음과 같다.

# In[46]:


def addM(A, B):
    assert shape(A) == shape(B)
    
    m, n = shape(A)
    
    return make_matrix(m, n, lambda i, j: A[i][j] + B[i][j])

def subtractM(A, B):
    assert shape(A) == shape(B)
    
    m, n = shape(A)
    
    return make_matrix(m, n, lambda i, j: A[i][j] - B[i][j])


# In[47]:


C = [[1, 3, 7],
     [1, 0, 0]]

D = [[0, 0, 5], 
     [7, 5, 0]]


# In[48]:


addM(C, D)


# In[49]:


subtractM(C, D)


# **행렬의 스칼라 곱셈**
# 
# 숫자 하나와 행렬의 곱셈을 행렬 스칼라 곱셈이라 부른다. 
# 스칼라 곱셈은 행렬의 각 항목을 지정된 숫자로 곱해 새로운 행렬을 생성한다.
# 즉, 벡터의 스칼라 곱셈과 동일한 방식이다. 
# 
# 예를 들어, (2, 3) 모양의 행렬의 스칼라 곱셈은 다음과 같다.
# 
# $$
# \begin{align*}
# 2\cdot 
# \begin{bmatrix}1&8&-3\\4&-2&5\end{bmatrix}
# &= \begin{bmatrix}2\cdot 1&2\cdot 8&2\cdot -3\\2\cdot 4&2\cdot -2&2\cdot 5\end{bmatrix} \\[.5ex]
# &= \begin{bmatrix}2&16&-6\\8&-4&10\end{bmatrix}
# \end{align*}
# $$

# **행렬 곱셈**
# 
# $m \times n$ 행렬 $A$와 $n \times p$ 행렬 $B$의 곱은 $m \times p$ 행렬이며, 
# 각 $(i, j)$번째 항목은 다음과 같이 정의된다.

# $$
# \begin{align*}
# (A B)_{ij}
# &= \sum _{k=0}^{n-1} A_{ik} \cdot B_{kj} \\
# &= A_{i0} \cdot B_{0j} + A_{i2} \cdot B_{2j} + \cdots + A_{i(n-1)} \cdot B_{(n-1)j}
# \end{align*}
# $$

# 그림으로 나타내면 다음과 같다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/datapy/master/jupyter-book/images/Matrix_mult_diagram.png" width="30%"></div>
# 
# 출처: [위키백과](https://en.wikipedia.org/wiki/Dot_product)

# 즉, 좌측 행렬 열의 수 $n$과 우측 행렬 행의 수 $n$이 같은 경우에만 곱셈이 가능하다.
# 예를 들어, $2 \times 3$ 행렬과 $3 \times 2$ 행렬의 곱셈은 다음과 같다.
# 
# $$
# \begin{align*}
# \begin{bmatrix}
#     1&0&2\\-1&3&1
# \end{bmatrix}
# \begin{bmatrix}
#     3&1\\2&1\\1&0
# \end{bmatrix}
# &=
# \begin{bmatrix}
#     (1\cdot 3+0\cdot 2+2\cdot 1)&(1\cdot 1+0\cdot 1+2\cdot 0)\\(-1\cdot 3+3\cdot 2+1\cdot 1)&(-1\cdot 1+3\cdot 1+1\cdot 0)
# \end{bmatrix} \\[.5ex]
# &= 
# \begin{bmatrix}
#     5&1\\4&2
# \end{bmatrix}
# \end{align*}
# $$

# **행렬 곱셈과 벡터 내적**
# 
# 행렬 $A B$의 $(i, j)$번째 항목 $(A B)_{ij}$는
# 벡터 내적과 깊이 연관되어 있다.
# 
# 먼저, $m \times n$ 행렬 $A$와 $n \times p$ 행렬 $B$의 곱 $A \times B$의
# $(i, j)$ 번째 항목은 다음과 같다.

# $$
# \begin{align*}
# (A B)_{ij}
# &= \sum _{k=0}^{n-1} A_{ik} \cdot B_{kj} \\
# &= A_{i0} \cdot B_{0j} + A_{i2} \cdot B_{2j} + \cdots + A_{i(n-1)} \cdot B_{(n-1)j}
# \end{align*}
# $$

# 반면에 행렬 $A$의 $i$ 행벡터를 
# 
# $$A^0_i = (A_{i0},\dots, A_{i(n-1)}),$$
# 
# 행렬 $B$의 $j$ 열벡터를 
# 
# $$B^1_j = (B_{0j},\dots, B_{(n-1)j})$$
# 
# 라 할 때, 다음이 성립한다.
# 
# $$
# A^0_i \cdot B^1_j
# = A_{i0} \cdot B_{0j} + A_{i2} \cdot B_{2j} + \cdots + A_{i(n-1)} \cdot B_{(n-1)j}
# $$
# 
# 즉, $(A B)_{ij}$는 $A$의 $i$ 행벡터와 $B$의 $j$ 열벡터의 **내적**으로 정의된다.

# 예를 들어 $2 \times 3$ 행렬과 $3 \times 2$ 행렬의 곱셈을 다시 살펴보자.

# $$
# \begin{align*}
# \begin{bmatrix}
#     1&0&2\\-1&3&1
# \end{bmatrix}
# \cdot 
# \begin{bmatrix}
#     3&1\\2&1\\1&0
# \end{bmatrix}
# &=
# \begin{bmatrix}
#     (1\cdot 3+0\cdot 2+2\cdot 1)&(1\cdot 1+0\cdot 1+2\cdot 0)\\(-1\cdot 3+3\cdot 2+1\cdot 1)&(-1\cdot 1+3\cdot 1+1\cdot 0)
# \end{bmatrix} \\[.5ex]
# &= 
# \begin{bmatrix}
#     5&1\\4&2
# \end{bmatrix}
# \end{align*}
# $$

# 위 계산을 벡터 내적으로 표현해보자.
# 먼저 두 행렬 $A$와 $B$를 지정한다.

# $$
# A = \begin{bmatrix}
#     1&0&2\\-1&3&1
# \end{bmatrix} \text{,}
# \qquad
# B = \begin{bmatrix}
#     3&1\\2&1\\1&0
# \end{bmatrix}
# $$

# 그러면 
# 
# $$(A\cdot B)_{11} = -1\cdot 1+3\cdot 1+1\cdot 0 = 2$$
# 
# 이고, 이것은 아래 결과와 동일하다.

# $$
# \begin{align*}
# A^0_1 \cdot B^1_1
# &= 
# \begin{bmatrix}
#     -1&3&1
# \end{bmatrix}
# \cdot
# \begin{bmatrix}
#     1 & 1 & 0
# \end{bmatrix} \\[.5ex]
# &=
# -1\cdot 1+3\cdot 1+1\cdot 0 \\
# &=
# 2
# \end{align*}
# $$

# **항등원**
# 
# 영행렬은 행렬 덧셈의 항등원이며, 단위행렬은 행렬 곱셈의 항등원이다.

# $$
# \begin{align*}
# \begin{bmatrix}
#     3&1 \\
#     2&1 \\
#     1&0
# \end{bmatrix}
# +
# \begin{bmatrix}
#     0&0 \\ 
#     0&0 \\
#     0&0
# \end{bmatrix}
# &= 
# \begin{bmatrix}
#     (3+0)&(1+0)\\
#     (2+0)&(1+0)\\
#     (1+0)&(0+0)
# \end{bmatrix} \\[.5ex]
# &= 
# \begin{bmatrix}
#     3&1\\
#     2&1\\
#     1&0
# \end{bmatrix}\\[2ex]
# \begin{bmatrix}
#     3&1 \\
#     2&1 \\
#     1&0
# \end{bmatrix}
# \begin{bmatrix}
#     1&0 \\ 
#     0&1
# \end{bmatrix}
# &=
# \begin{bmatrix}
#     (3\cdot 1+1\cdot 0)&(3\cdot 0+1\cdot 1) \\
#     (2\cdot 1+1\cdot 0)&(2\cdot 0+1\cdot 1) \\
#     (1\cdot 1+0\cdot 0)&(1\cdot 0+0\cdot 1) \\
# \end{bmatrix} \\[.5ex]
# &= 
# \begin{bmatrix}
#     3&1\\
#     2&1\\
#     1&0
# \end{bmatrix}
# \end{align*}
# $$

# **전치행렬**

# 행렬의 **전치**란 행과 열을 바꾸는 것으로, 행렬 $A$의 전치는 $A^T$로 나타낸다. 
# 즉, $A$가 $m \times n$ 행렬이면 $A^T$는 $n \times m$ 행렬이며,
# 그리고 $A^T$의 $i$행의 $j$열번째 값은 $A$의 $j$행의 $i$열번째 값이다. 
# 즉,
# 
# $$
# A ^{T}_{ij} = A_{ji}
# $$

# 예를 들어, $2\times 3$ 행렬의 전치는 $3 \times 2$ 행렬이 되며 다음과 같이 작동한다.
# 
# $$
# \begin{bmatrix}
#     9&8&7\\
#     -1&3&4
# \end{bmatrix}^{T}
# =
# \begin{bmatrix}
#     9&-1\\
#     8&3\\
#     7&4
# \end{bmatrix}
# $$

# **전치행렬의 성질**

# $a$를 스칼라, $A, B$를 크기가 같은 행렬이라 하자. 이때 다음이 성립한다.
# 
# * $(A^T)^T = A$
# * $(A + B)^T = A^T + B^T$
# * $(A - B)^T = A^T - B^T$
# * $(a\cdot A)^T = a\cdot A^T$
# * $(A B)^T = B^T A^T$

# ## 연습문제

# 참고: [(실습) 선형대수 기초 밑바닥부터](https://colab.research.google.com/github/codingalzi/datapy/blob/master/practices/practice-linear_algebra_basics.ipynb)
