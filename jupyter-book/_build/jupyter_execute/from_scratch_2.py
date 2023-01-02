#!/usr/bin/env python
# coding: utf-8

# (sec:from_scratch_2)=
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
# 아래 코드는 {ref}`자료형 힌트 <sec:from_scratch_1>`와
# [리스트 조건제시법](https://codingalzi.github.io/pybook/collections2.html#sec-comprehension)을 
# 활용한다. 

# ## 벡터

# 벡터는 유한 개의 숫자를 담고 있으며, 벡터의 길이를 **차원**(dimension)이라 부른다.
# 
# __주의사항:__ 넘파이 어레이의 차원과 다른 개념임에 주의하라.
# 
# 벡터는 수학과 통계에서 많이 사용된다.
# 
# * 수학 예제: 2차원 평면 공간에서 방향과 크기를 표현하는 2차원 벡터 `(x, y)`
# 
# * 통계 예제: 사람들의 키, 몸무게, 나이로 이루어진 3차원 벡터 `(키, 몸무게, 나이)`
# 
# * 통계 예제: 네 번의 시험 점수로 이루어진 4차원 벡터 `(1차점수, 2차점수, 3차점수, 4차점수)`

# **벡터 자료형**

# 벡터 자료형은 부동소수점들로 이루어진 리스트로 정의될 수 있다. 

# In[1]:


from typing import List

Vector = List[float]


# * 예제: 키, 몸무게, 나이로 구성된 3차원 벡터

# In[2]:


# (키, 몸무게, 나이)

height_weight_age1 : Vector = [70, 170, 50]
height_weight_age2 : Vector = [66, 163, 50]


# * 예제: 1차부터 4차까지의 시험 점수로 구성된 4차원 벡터

# In[3]:


# (1차점수, 2차점수, 3차점수, 4차점수)

grades1 : Vector = [95, 80, 75, 62]
grades2 : Vector = [85, 82, 79, 82]


# **벡터 덧셈**
# 
# 차원이 같은 벡터 두 개의 덧셈은 같은 위치에 있는 항목끼기 더한 결과로 이루어진 벡터를 생성한다.

# $$
# \begin{align*}
# (1, 2) + (2, 1) &= (1+2, 2+1) \\
# &= (3, 3) \\[1ex]
# (6, 3, 2) + (1, 7, 9) & = (6+1, 3+7, 2+9) \\
# &= (7, 10, 11)
# \end{align*}
# $$

# **벡터 덧셈의 기하적 의미**
# 
# 벡터 $a$와 벡터 $b$의 합 $a+b$의 의미를 아래 그래프에서처럼 해석할 수 있다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/datapy/master/jupyter-book/images/vector_addition.png" width="40%"></div>
# 
# 출처: [위키백과](https://en.wikipedia.org/wiki/Euclidean_vector)

# **벡터 덧셈 함수**
# 
# 차원이 같은 두 벡터의 항목별 합을 항목으로 같은 벡터를 반환하는 함수는 다음과 같다.

# In[4]:


def addV(v: Vector, w: Vector) -> Vector:
    assert len(v) == len(w)   # 두 벡터의 길이가 같아야 함

    return [v_i + w_i for v_i, w_i in zip(v, w)]


# In[5]:


addV(height_weight_age1, height_weight_age2)


# In[6]:


addV(grades1, grades2)


# **벡터 덧셈 함수 일반화**
# 
# 동일한 차원의 임의의 개수의 벡터를 더하는 함수를 다음과 같이 정의할 수 있다.

# In[7]:


def vector_sum(vectors: List[Vector]) -> Vector:
    """
    인자: 동일한 차원의 벡터들의 리스트
    반환값: 각 항목의 합으로 이루어진 동일한 차원의 벡터
    """
    
    assert vectors                   # 1개 이상의 벡터가 주어져야 함

    num_elements = len(vectors[0])   # 벡터 개수
    
    assert all(len(v) == num_elements for v in vectors)   # 모든 벡터의 크기가 같아야 함

    # 동일한 위치의 항목을 모두 더한 값들로 이루어진 벡터 반환
    return [sum(vector[i] for vector in vectors)
            for i in range(num_elements)]


# 예를 들어, 2차원 벡터 네 개를 더한 결과는 다음과 같다.

# In[8]:


vector_sum([[1, 2], [3, 4], [5, 6], [7, 8]])


# **벡터 뺄셈**
# 
# 차원이 같은 벡터 두 개의 덧셈은 같은 위치에 있는 항목끼기 뺀 결과로 이루어진 벡터를 생성한다.

# $$
# \begin{align*}
# (1, 2) - (2, 1) &= (1-2, 2-1) \\
# &= (-1, 1) \\[1ex]
# (6, 3, 2) - (1, 7, 9) & = (6-1, 3-7, 2-9) \\
# &= (5, -4, -7)
# \end{align*}
# $$

# **벡터 뺄셈의 기하적 의미**
# 
# 벡터 $a$와 벡터 $b$의 합 $a-b$의 의미를 아래 그래프에서처럼 해석할 수 있다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/datapy/master/jupyter-book/images/vector_subtraction.png" width="20%"></div>
# 
# 출처: [위키백과](https://en.wikipedia.org/wiki/Euclidean_vector)

# **벡터 뺄셈 함수**
# 
# 차원이 같은 두 벡터의 항목별 차를 항목으로 같은 벡터를 반환하는 함수는 다음과 같다.

# In[9]:


def subtractV(v: Vector, w: Vector) -> Vector:
    assert len(v) == len(w)   # 두 벡터의 길이가 같아야 함

    return [v_i - w_i for v_i, w_i in zip(v, w)]


# In[10]:


subtractV(height_weight_age1, height_weight_age2)


# In[11]:


subtractV(grades1, grades2)


# **벡터 스칼라 곱셈**
# 
# 숫자 하나와 벡터의 곱셈을 스칼라 곱셈이라 부른다. 
# 스칼라 곱셈은 벡터의 각 항목을 지정된 숫자로 곱해 새로운 벡터를 생성한다. 

# $$
# \begin{align*}
# 3\cdot (1, 2) &= (3\cdot 1, 3\cdot 2) \\
# &= (3, 6) \\[1ex]
# 2\cdot (6, 3, 2) & = (2\cdot 6, 2\cdot 3, 2\cdot 2) \\
# &= (12, 6, 4)
# \end{align*}
# $$

# **스칼라 곱셈의 기하적 의미**
# 
# <table>
# <tr>
#     <td><img src="https://raw.githubusercontent.com/codingalzi/datapy/master/jupyter-book/images/Scalar_mult_3.png" width="60%"></td>
#     <td><img src="https://raw.githubusercontent.com/codingalzi/datapy/master/jupyter-book/images/Scalar_mult_2.png" width="60%"></td>
# </tr>
# </table>
# 
# 출처: [위키백과](https://en.wikipedia.org/wiki/Euclidean_vector)

# **벡터 스칼라 곱셈 함수**
# 
# 벡터의 각 항목에 동일한 부동소수점을 곱한 결과를 반화하는 함수는 다음과 같다.

# In[12]:


def scalar_multV(c: float, v: Vector) -> Vector:
    return [c * v_i for v_i in v]


# In[13]:


scalar_multV(2, [1, 2, 3])


# **항목별 평균 벡터**
# 
# 동일한 차원의 여러 벡터가 주어졌을 때 항목별 평균을 구할 수 있다.
# 항목별 평균은 항목끼리 모두 더한 후 벡터의 개수로 나눈다.
# 따라서 벡터의 덧셈과 스칼라 곱셈을 이용할 수 있다.

# $$
# \begin{align*}
# \frac 1 3 \cdot \left ((1, 2) + (2, 1) + (2, 3) \right) 
# &=  \frac 1 3 \cdot (1+2+2, 2+1+3) \\
# &= \left (\frac 5 3, 2 \right)
# \end{align*}
# $$

# **항목별 평균 벡터 함수**
# 
# 동일한 차원의 여러 벡터가 주어졌을 때 항목별 평균으로 이루어진 벡터를 반환하는 함수는 다음과 같다.

# In[14]:


def vector_mean(vectors: List[Vector]) -> Vector:
    n = len(vectors)
    return scalar_multV(1/n, vector_sum(vectors))


# **참고:** 5/3은 약 1.666666이다. 

# In[15]:


vector_mean([[1, 2], [2, 1], [2, 3]])


# **벡터 내적**
# 
# 차원이 같은 벡터 두 개의 내적은 같은 위치에 있는 항목끼기 곱한 후 모두 더한 값이다.
# 벡터 $u = (u_1, \cdots, u_n)$와 벡터 $v = (v_1, \cdots, v_n)$가 주어졌을 때
# 두 벡터의 내적은 아래와 같다. 
# 
# $$
# \begin{align*}
# u \cdot v &= \sum_{i=1}^n u_i\cdot v_i \\
# &= u_1\cdot v_1 + \cdots + u_n\cdot v_n
# \end{align*}
# $$

# **벡터 내적의 기하적 의미**
# 
# 두 개의 벡터 $A$와 $B$가 주어졌고, 벡터 $B$의 길이가 1이라고 가정하자.
# 그러면 내적 $A \cdot B$는 벡터 $A$가 벡터 $B$ 방향으로 사영되었을 때의 길이를 나타낸다. 

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/datapy/master/jupyter-book/images/dot_product.png" width="20%"></div>
# 
# 출처: [위키백과](https://en.wikipedia.org/wiki/Dot_product)

# 임의의 벡터 $B$에 대한 내적 $A \cdot B$는 다음과 같다. ($\theta$ 두 벡터가 이루는 각을 나타낸다.)
# 
# $$
# A \cdot B = \vert A\vert\cdot \vert B\vert \cdot \cos\theta
# $$

# **벡터 내적 함수**
# 
# 동일 차원의 두 벡터의 내적을 반환하는 함수는 다음과 같다.

# In[16]:


def dot(v: Vector, w: Vector) -> float:
    assert len(v) == len(w), "벡터들의 길이가 동일해야 함"""

    return sum(v_i * w_i for v_i, w_i in zip(v, w))


# In[17]:


dot([1, 2, 3], [4, 5, 6]) == 32


# **벡터의 크기**
# 
# 벡터 $v = (v_1, \cdots, v_n)$가 주어졌을 때 각 항목별 제곱의 합은 $v$와 $v$ 자신의 내적과 같다.
# 
# $$v \cdot v = v_1^2 + \cdots + v_n^2$$
# 
# 그런데 이 값은 정확하게 벡터 $v$의 크기 $\vert v\vert$의 제곱이다. 
# 따라서 다음이 성립한다. 
# 
# $$\vert v\vert = \sqrt{v \cdot v} = \sqrt{v_1^2 + \cdots + v_n^2}$$
# 
# 예를 들어 벡터 $(3, 4)$의 크기는 다음과 같다.
# 
# $$\vert (3, 4)\vert  = \sqrt{3^2 + 4^2} = \sqrt{5^2} = 5$$

# In[18]:


import math

def sum_of_squares(v: Vector) -> float:

    return dot(v, v)

def magnitude(v: Vector) -> float:
    return math.sqrt(sum_of_squares(v))


# In[19]:


magnitude([3, 4])


# **벡터 사이의 거리**
# 
# 벡터 $v = (v_1, \cdots, v_n)$와 벡터 $w = (w_1, \cdots, w_n)$ 사이의 거리는 
# 벡터 $v-w$의 크기, 즉 아래처럼 정의된 $\vert v - w \vert$이다.
# 
# $$
# \begin{align*}
# \vert v - w\vert &= \sqrt{(v-w) \cdot (v-w)} \\[.5ex]
# &= \sqrt{(v_1-w_1)^2 + \cdots + (v_n-w_n)^2}
# \end{align*}
# $$
# 
# 
# 예를 들어, 두 벡터 $(1, 2)$와 $(2, 1)$ 사이의 거리는 다음과 같다.
# 
# $$\vert (1, 2) - (2, 1)\vert  = \sqrt{(-1)^2 + 1^2} = \sqrt{2}$$

# In[20]:


def squared_distance(v: Vector, w: Vector) -> float:
    return sum_of_squares(subtractV(v, w))

# 버전 1
def distance(v: Vector, w: Vector) -> float:
    return math.sqrt(squared_distance(v, w))

# 버전 2
def distance(v: Vector, w: Vector) -> float:
    return magnitude(subtractV(v, w))


# **참고:** $\sqrt{2} = 1.41421356237...$

# In[21]:


distance([1,2], [2,1])


# ## 행렬

# 행렬(matrix)은 보통 숫자들을 직사각형 형태로 배열한 것이다. 
# 
# 예를 들어, $1, 2, 3, 4, 5, 6$ 여섯 개의 항목을 가진 
# 행렬의 모양(shape)은 네 종류가 있다. 
# 이유는 6을 두 개의 양의 정수의 곱셈으로 표현하는 방법이 네 가지이기 때문이다. 
# 
# $$ 6 = 1\times6 = 2\times 3 = 3 \times 2 = 6 \times 1$$

# * $1\times 6$ 행렬: 한 개의 행과 여섯 개의 열
# 
# \begin{bmatrix}
#     1 & 2 & 3 & 4 & 5 & 6
# \end{bmatrix}

# * $2 \times 3$ 행렬: 두 개의 행과 세 개의 열
# 
# \begin{bmatrix}
#     1 & 2 & 3\\
#     4 & 5 & 6
# \end{bmatrix}

# * $3 \times 2$ 행렬: 세 개의 행과 두 개의 열
# 
# \begin{bmatrix}
#     1 & 2 \\
#     3 & 4 \\
#     5 & 6
# \end{bmatrix}

# * $6 \times 1$ 행렬: 여섯 개의 행과 한 개의 열
# 
# \begin{bmatrix}
#     1 \\
#     2 \\
#     3 \\
#     4 \\
#     5 \\
#     6
# \end{bmatrix}

# **행렬 자료형**

# 행렬을 리스트의 리스트, 즉 2중 리스트로 구현한다.

# In[22]:


Matrix = List[List[float]]


# 예를 들어, 아래 `A`와 `B`는 각각 (2, 3), (3, 2) 모양의 행렬을 나타낸다.

# In[23]:


A = [[1, 2, 3],  # 2 x 3 행렬
     [4, 5, 6]]

B = [[1, 2],     # 3 x 2 행렬
     [3, 4],
     [5, 6]]


# **행렬의 모양(shape)**
# 
# $n$ 개의 행과 $k$ 개의 열로 구성된 행렬을 $n \times k$ 행렬이라 부르며,
# $(n,k)$를 해당 행렬의 모양(shape)라 부른다.
# 아래 함수 `shape()`는 주어진 행렬의 모양을 튜플로 반환한다.

# In[24]:


from typing import Tuple

def shape(A: Matrix) -> Tuple[int, int]:
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0   # number of elements in first row
    return num_rows, num_cols


# In[25]:


shape(A)


# In[26]:


shape(B)


# **행벡터와 열벡터**
# 
# 아래 두 함수는 각각 지정된 행와 지정된 열의 항목들로 구성된 행벡터와 열벡터를 반환한다.

# In[27]:


# 행벡터 계산
def get_row(A: Matrix, i: int) -> Vector:
    return A[i]             

# 열벡터 계산
def get_column(A: Matrix, j: int) -> Vector:
    return [A_i[j] for A_i in A]


# 행렬 `A`의 0번 행은 다음과 같다.

# In[28]:


get_row(A, 0)


# 행렬 `B`의 1번 열은 다음과 같다.

# In[29]:


get_column(B, 1)


# **행렬 생성 함수**
# 
# 아래 함수는 지정된 모양의 행렬을 생성한다.
# 셋째 인자는 지정된 위치의 항목을 계산하는 함수이다. 
# 
# * 인자
#     * `num_rows`: 행의 수
#     * `num_rows`: 열의 수
#     * `entry_fn`: i, j가 주어지면 i행, j열에 위치한 값 계산
# * 반환값: 지정된 방식으로 계산된 (i, j) 모양의 행렬

# In[30]:


from typing import Callable

def make_matrix(num_rows: int,
                num_cols: int,
                entry_fn: Callable[[int, int], float]) -> Matrix:
    return [ [entry_fn(i, j) for j in range(num_cols)] for i in range(num_rows)]   


# **영행렬**
# 
# 영행렬(zero matrix)이란 행렬의 모든 원소의 값이 0인 행렬을 말한다.
# 예를 들어 아래 행렬은 (3, 2) 모양의 영행렬이다.
# 
# $$
# \begin{bmatrix}
#     0 & 0 \\
#     0 & 0 \\
#     0 & 0
# \end{bmatrix}
# $$

# 지정된 모양의 영행렬을 생성하는 함수는 다음과 같다.

# In[31]:


def zero_matrix(n: int, m:int) -> Matrix:
    return make_matrix(n, m, lambda i, j: 0)


# In[32]:


zero_matrix(5,7)


# **단위행렬**
# 
# 단위행렬(identity matrix)은 정사각형 모양의 행렬 중에서 대각선 상에 위치한 항목은 1이고
# 나머지는 0인 행렬을 말한다. 
# 예를 들어 아래 행렬은 (5, 5) 모양의 단위행렬이다.
# 
# \begin{bmatrix}
#     1&0&0&0&0 \\
#     0&1&0&0&0 \\
#     0&0&1&0&0 \\
#     0&0&0&1&0 \\
#     0&0&0&0&1
# \end{bmatrix}
# 
# 단위행렬은 행과 열의 개수가 동일한 정방행렬이며,
# 지정된 모양의 단위행렬을 생성하는 함수는 다음과 같다.

# In[33]:


def identity_matrix(n: int) -> Matrix:
    return make_matrix(n, n, lambda i, j: 1 if i == j else 0)


# In[34]:


identity_matrix(5)


# **행렬 덧셈과 뺄셈**
# 
# 모양이 같은 두 행렬의 덧셈/뺄셈은 항목별로 더한/뺀 결과로 이루어진 행렬이다. 
# 즉, 벡터의 덧셈/뺄셈과 동일한 방식이다.
# 예를 들어, $2 \times 3$ 행렬의 덧셈/뺄셈은 다음과 같다.

# $$
# \begin{align*}
# \begin{bmatrix}1&3&7\\1&0&0\end{bmatrix} 
# + \begin{bmatrix}0&0&5\\7&5&0\end{bmatrix}
# &= \begin{bmatrix}1+0&3+0&7+5\\1+7&0+5&0+0\end{bmatrix} \\[.5ex]
# &= \begin{bmatrix}1&3&12\\8&5&0\end{bmatrix}\\[2ex]
# \begin{bmatrix}1&3&7\\1&0&0\end{bmatrix} 
# - \begin{bmatrix}0&0&5\\7&5&0\end{bmatrix}
# &= \begin{bmatrix}1-0&3-0&7-5\\1-7&0-5&0-0\end{bmatrix} \\[.5ex]
# &= \begin{bmatrix}1&3&2\\-6&-5&0\end{bmatrix}
# \end{align*}
# $$

# 행렬의 덧셈과 뺄셈을 계산하는 함수는 다음과 같다.

# In[35]:


def addM(A: Matrix, B: Matrix) -> Matrix:
    assert shape(A) == shape(B)
    
    m, n = shape(A)
    
    return make_matrix(m, n, lambda i, j: A[i][j] + B[i][j])

def subtractM(A: Matrix, B: Matrix) -> Matrix:
    assert shape(A) == shape(B)
    
    m, n = shape(A)
    
    return make_matrix(m, n, lambda i, j: A[i][j] - B[i][j])


# In[36]:


C = [[1, 3, 7],
     [1, 0, 0]]

D = [[0, 0, 5], 
     [7, 5, 0]]


# In[37]:


addM(C, D)


# In[38]:


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
