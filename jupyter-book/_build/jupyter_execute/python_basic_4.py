#!/usr/bin/env python
# coding: utf-8

# (sec:python_basic_4)=
# # 제어문

# 프로그램 실행의 흐름을 제어하는 명령문을 소개한다.
# 
# * `if` 조건문
# * `for` 반복문
# * `while` 반복문

# ## `if` 조건문

# `if` 다음에 위치한 조건식이 참이면 해당 본문 불록의 코드를 실행한다.

# In[1]:


x = -2

if x < 0:
    print("It's negative")


# __주의사항:__ "It's negative" 문자열 안에 작은따옴표가 
# 사용되기 때문에 반드시 큰따옴표로 감싸야 한다. 

# 조건식이 만족되지 않으면 해당 본문 블록을 건너뛴다.

# In[2]:


x = 4

if x < 0:
    print("It's negative")
    
print("if문을 건너뛰었음!")


# 경우에 따른 여러 조건을 사용할 경우 원하는 만큼의 `elif` 문을 사용하고
# 마지막에 `else` 문을 한 번 사용할 수 있다. 
# 하지만 `else` 문이 생략될 수도 있다.
# 
# 위에 위치한 조건식의 만족여부부터 조사하며 한 곳에서 만족되면 나머지 부분은 무시된다.

# In[3]:


if x < 0:
    print('음수')
elif x == 0:
    print('숫자 0')
elif 0 < x < 5:
    print('5 보다 작은 양수')
else:
    print('5 보다 큰 양수')


# __참고:__ 부울 연산자가 사용되는 경우에도 하나의 표현식이 참이거나 거짓이냐에 따라
# 다른 표현식을 검사하거나 무시하기도 한다.
# 예를 들어, `or` 연산자는 첫 표현식이 `True` 이면 다른 표현식은 검사하지 않는다.
# 아래 코드에서 `a < b`가 참이기에 `c/d > 0` 은 아예 검사하지 않는다.
# 하지만 `c/d > 0`을 검사한다면 오류가 발생해야 한다.
# 이유는 0으로 나누는 일은 허용되지 않기 때문이다.

# In[4]:


a = 5; b = 7
c = 8; d = 0
if a < b or c / d > 0:
    print('오른편 표현식은 검사하지 않아요!')


# 실제로 0으로 나눗셈을 시도하면 `ZeroDivisionError` 라는 오류가 발생한다.

# ```python
# In [13]: c / d > 0
# ---------------------------------------------------------------------------
# ZeroDivisionError                         Traceback (most recent call last)
# <ipython-input-13-2fb2320cb664> in <module>
# ----> 1 c / d > 0
# 
# ZeroDivisionError: division by zero
# ```

# ### `pass` 명령문

# 아무 것도 하지 않고 다음으로 넘어가도는 하는 명령문이다. 
# 주로 앞으로 채워야 할 부분을 명시할 때 또는
# 무시해야 하는 경우를 다룰 때 사용한다.
# 
# 아래 코드는 x가 0일 때 할 일을 추후에 지정하도록 `pass` 명령문을 사용한다.

# In[5]:


x = 0

if x < 0:
    print('negative!')
elif x == 0:
    # 할 일: 추추 지정
    pass
else:
    print('positive!')


# ### 삼항 표현식

# 삼항 표현식 `if ... else ...` 를 이용하여 지정된 값을 한 줄로 간단하게 표현할 수 있다.
# 예를 들어, 아래 코드를 살펴보자.

# In[6]:


x = 5

if x >= 0:
    y = 'Non-negative'
else:
    y = 'Negative'
    
print(y)


# 변수 `y`를 아래와 같이 한 줄로 선언할 수 있다.

# In[7]:


y = 'Non-negative' if x >= 0 else 'Negative'

print(y)


# ## `for` 반복문

# `for` 반복문은 리스트, 튜플, 문자열과 같은 이터러블 자료형의 값에 포함된 항목을 순회하는 데에 사용된다.
# 기본 사용 양식은 다음과 같다.

# ```python
# for item in collection:
#     # 코드 블록 (item 변수 사용 가능)
# ```

# ### `continue` 명령문

# `for` 또는 아래에서 소개할 `while` 반복문이 실행되는 도중에
# `continue` 명령문을 만나는 순간 현재 실행되는 코드의 실행을 멈추고
# 다음 순번 항목을 대상으로 반복문을 이어간다.
# 
# 예를 들어, 리스트에 포함된 항목 중에서 `None`을 제외한 값들의 합을 계산할 수 있다.

# In[8]:


sequence = [1, 2, None, 4, None, 5]
total = 0

for value in sequence:
    if value is None:
        continue
    total += value
    
print(total)


# ### `break` 명령문

# `for` 또는 아래에서 소개할 `while` 반복문이 실행되는 도중에
# `break` 명령문을 만나는 순간 현재 실행되는 반복문 자체의 실행을 멈추고,
# 다음 명령을 실행한다. 
# 
# 예를 들어, 리스트에 포함된 항목들의 합을 계산하는 과정 중에 5를 만나는 순간 
# 계산을 멈추게 하려면 다음과 같이 `break` 명령문을 이용한다.

# In[9]:


sequence = [1, 2, 0, 4, 6, 5, 2, 1]
total_until_5 = 0

for value in sequence:
    if value == 5:
        break
    total_until_5 += value
    
print(total_until_5)


# `break` 명령문은 가장 안쪽에 있는 `for` 또는 `while` 반복문을 빠져나가며,
# 또 다른 반복문에 의해 감싸져 있다면 해당 반복문을 이어서 실행한다.
# 예를 들어, 아래 코드는 0, 1, 2, 3으로 이루어진 순서쌍들을 출력한다.
# 그런데 둘째 항목이 첫째 항목보다 큰 경우는 제외시킨다.
# 
# __참고:__ `range(4)`는 리스트 `[1, 2, 3, 4]`와 유사하게 작동한다.
# 레인지(range)에 대해서는 잠시 뒤에 살펴본다.

# In[10]:


for i in range(4):
    for j in range(4):
        if j > i:
            break
        print((i, j))


# 리스트, 튜를 등의 항목이 또 다른 튜플, 리스트 등의 값이라면 아래와 같이 여러 개의 
# 변수를 이용하여 `for` 반복문을 실행할 수 있다.

# In[11]:


an_iterator = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]

for a, b, c in an_iterator:
    print(a * (b + c))


# 위 코드에서 `a`, `b`, `c` 각각은 길이가 3인 튜플의 첫째, 둘째, 셋째 항목을 가리킨다. 
# 따라서 위 결과는 아래 계산의 결과를 보여준 것이다.
# 
# ```python
# 1 * (2 + 3) = 5
# 4 * (5 + 6) = 44
# 7 * (8 + 9) = 119
# ```

# ## `while` 반복문

# 지정된 조건이 만족되는 동안, 또는 실행 중에 `break` 명령문을 만날 때까
# 동일한 코드를 반복실행 시킬 때 `while` 반복문을 사용한다.
# 
# 아래 코드는 256부터 시작해서 계속 반으로 나눈 값을 더하는 코드이며,
# 나누어진 값이 0보다 작거나 같아지면, 또는 더한 합이 500보다 커지면 바로 실행을 멈춘다.

# In[12]:


x = 256
total = 0

while x > 0:
    if total > 500:
        break
    total += x
    x = x // 2
    
print(total)

