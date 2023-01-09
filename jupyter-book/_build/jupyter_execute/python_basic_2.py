#!/usr/bin/env python
# coding: utf-8

# (sec:python_basic_2)=
# # 기초 문법

# ## 구문

# 구문 또는 신택스<font size='2'>syntax</font>는 프로그래밍 언어에서 정해진 문법에 따라 프로그램의 형태와 구조를 지정하는 것을 가리킨다.
# 파이썬의 구문은 C, Java 등의 그것과는 다르다. 예를 들어, 중괄호(`{ }`)와 세미콜론(`;`)을 사용하지 않으며,
# 대신에 들여쓰기와 줄바꿈을 기본으로 사용한다.

# **들여쓰기**

# 들여쓰기를 이용하여 명령문 블록<font size='2'>block</font>을 지정한다. 
# 예를 들어 `for` 반복문의 본문에 `if ... else ...` 조건문을 
# 사용하고자 할 경우 아래와 같이 작성한다.

# In[1]:


data = [3, 1, 5, 2, 4]
pivot = data[0]
less = []
greater = []

for x in data:
    if x < pivot:         # for 본문 들여쓰기 사용
        less.append(x)    # if 본문 들여쓰기 사용   
    else:                 
        greater.append(x) # else 본문 들여쓰기 사용

print(f"less: {less}")
print(f"greater: {greater}")


# :::{admonition} <kbd>Tab</kbd> 키 활용
# 
# 코딩할 때 <kbd>Tab</kbd> 키는 두 가지 기능으로 유용하게 사용된다.
# 
# 첫째, **들여쓰기**. 커서가 줄 맨 왼쪽에 위치했을 때 <kbd>Tab</kbd> 키를 누르면 
# 보통 스페이스 네 개에 해당하는 만큼 들여쓰기가 이뤄진다.
# 대부분의 편집기는 <kbd>Enter</kbd> 키를 치면 들여쓰기를 적절하게 대신 실행하지만
# 사용자가 직접 <kbd>Tab</kbd> 키를 이용하여 들여쓰기 정도를 조절해야 할 필요가 있다.
# 들여쓰기를 반대로 실행하려면 <kbd>Shift</kbd> + <kbd>Tab</kbd> 조합을 이용한다.
# 
# 둘째, **탭 완성**. 함수, 변수, 객체 등을 작성하다가 <kbd>Tab</kbd> 키를 누르면
# 이름을 자동으로 완성해준다. 
# 대부분의 편집키는 또한 탭키를 누르면 적절한 함수, 변수, 객체를 추천하기도 한다.
# <kbd>Tab</kbd> 키는 보다 빠른 코딩을 지원하는 것 뿐만 아니라, 
# 코드 작성 오타 또는 오류를 줄여 준다.
# :::

# **줄바꿈**

# 하나의 명령문은 한 줄에 작성하는 것이 원칙이다.
# 아래 코드는 세 개의 변수를 각각 한 줄에 하나씩 선언하는 것을 보여준다.

# In[2]:


a = 5
b = 6
c = 7


# 연속된 변수 할당의 경우 파이썬은 아래 방식도 지원한다.

# In[3]:


a, b, c = 5, 6, 7
print(f"a: {a}", f"b: {b}", f"c: {c}", sep='\n')


# **주의사항:** `f"a: {a}, ..."`는 **f-문자열**, 즉 변수를 포함하는 문자열을
# 생성하는 **문자열 포매팅**<font size='2'>string formatting</font>이다.
# 문자열 포매팅에 대한 보다 자세한 설명은 [파이썬 프로그래밍 기초: 문자열 포매팅](https://codingalzi.github.io/pybook/datatypes.html?highlight=format#id10)를 참고한다.

# ## 주석

# 샵 기호(`#`) 다음에 오는 문장은 파이썬 인터프리터에의 의해 무시된다.
# 주로 코드의 일부 기능을 잠시 해제하거나 코드에 대한 설명을 전달하는 주석으로 활용된다.
# 예를 들어 아래 코드에서 주석은 코드 일부를 실행 해제하는 기능으로 사용되었다.

# In[4]:


items = ["라디오", "", "tv", "전화"]
results = []

for item in items:
    # if len(item) == 0:  # 코드 실행 해제
    #   continue          # 코드 실행 해제
    results.append(item)

print(results)


# 위 코드의 주석을 제거하면 다른 결과가 나온다.
# 이유는 `if` 조건문의 본문 코드인 `continue`에 의해 빈 문자열인 경우
# 무시되기 때문이다.

# In[5]:


items = ["라디오", "", "tv", "전화"]
results = []

for item in items:
    if len(item) == 0:
        continue
    results.append(item)

print(results)


# 아래와 같이 명령문 끝 부분에 주석을 달아 
# 해당 명령문에 대한 설명 또는 정보를 제공하기도 한다.

# In[6]:


print("여기까지 공부했습니다.") # 여기까지 진행한 것을 확인해주는 문장 출력


# ## 변수

# **참조**

# 변수가 가리키는 값이 리스트와 같이 좀 복잡한 객체일 때는 파이썬 실행기 내부에서
# 변수가 해당 값을 **참조**(reference)한다.
# 참조는 변수가 단순히 하나의 값을 가리키는 것 이외에 부가적인 기능도 수행한다.
# 
# 예를 들어 변수 `a`가 리스트 `[1, 2, 3]`을 참조하도록 하자.

# In[7]:


a = [1, 2, 3]


# 그리고 변수 `b`를 다음과 같이 선언하면 변수 `a`와 동일한 값을 참조한다. 

# In[8]:


b = a


# 아래 그림에서에서처럼 `a`와 `b`가 화살표로 동일한 리스트를 가리키는 방식으로
# 참조를 표현할 수 있다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/datapy/master/jupyter-book/images/variables-a-b-1.png" style="width:340px;"></div>

# `a`와 `b`가 동일한 값을 참조하기에 `a`가 참조하는 값을 변화시키면 `b`도 영향을 받는다.
# 아래 코드는 `a`가 가리키는 리스트에 항목을 추가하면 `b`가 동일한 리스트를 가리키기에
# `b`가 가리키는 값도 함께 변하는 것을 보여준다.

# In[9]:


a.append(4) # 항목 추가
b


# 그림으로 표현하면 다음과 같다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/datapy/master/jupyter-book/images/variables-a-b-2.png" style="width:340px;"></div>

# 반면에 정수와 같이 보다 간단한 객체를 변수에 할당하는 경우는 다르게 작동한다.

# In[10]:


a = 4
b = a


# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/datapy/master/jupyter-book/images/variables-a-b-4.png" style="width:360px;"></div>

# 변수 `a`가 가리키는 값을 변경하더라도 변수 `b`가 가리키는 값은 영향을 받지 않는다.

# In[11]:


a = a + 1

print(f"a = {a}", f"b = {b}", sep="\n")


# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/datapy/master/jupyter-book/images/variables-a-b-5.png" style="width:360px;"></div>

# **스코프**

# 파이썬에서 다루는 모든 객체는 스코프<font size='2'>scope</font>라 불리는 자신만의 활동영역이 있으며
# 자신의 활동영역을 벗어나서는 존재 자체를 인정받지 못한다. 
# 이는 변수에게도 동일하게 적용되는데 특히 함수와 관련되어 전역 변수와 지역 변수로 서로 다른 활동영역을 갖는다.

# **전역 변수와 지역 변수**

# 특정 함수와 상관이 없이 선언된 변수인 **전역 변수**<font size='2'>global variables</font>는 어디에서든 사용할 수 있지만,
# 어떤 함수의 매개 변수 또는 함수 본문 내에서 선언된 변수인 **지역 변수**<font size='2'>local variables</font>는 함수 밖에서 사용할 수 없다.

# 아래 코드에서 `some_list`, `element`는 함수의 매개 변수로 선언된 지역 변수이고,
# `data_original`은 함수 본문에서 선언된 지역변수이다.

# In[12]:


def append_element(some_list, element):
    data_original = some_list
    data_original.append(element)
    return data_original


# 아래 코드에서 `data`는 함수 밖에서 선언된 전역 변수이며,
# 함수에 인자로 사용되었다.

# In[13]:


data = [1, 2, 3]

append_element(data, 4)


# 함수의 실행이 끝나도 `data`는 변형된 리스트를 가리키는 변수로 남아 있다.
# 또한 함수가 실행될 때 생성되는 `data_original`이 `data`와 동일한 리스트를 참조하기에
# 리스트에 4를 추가한 결과가 `data`가 가리키는 리스트에도 동일한 영향을 준다.

# In[14]:


data


# 반면에 지역 변수인 `data_original`은 함수 실행이 멈춘 후에는 더 이상 사용할 수 없다.
# 
# ```python
# In [30]: print(data_original)
# ---------------------------------------------------------------------------
# NameError                                 Traceback (most recent call last)
# <ipython-input-1-80a9247ee20e> in <module>
# ----> 1 print(data_original)
# 
# NameError: name 'data_original' is not defined
# ```

# ## 함수 호출과 인자

# **함수 호출**

# 함수를 적절한 인자와 함께 실행하는 것을 **함수 호출**<font size='2'>function call</font>이라 한다.
# 예를 들어, 아래와 같이 세 개의 인자를 받는 함수를 정의한다.

# In[15]:


def fun(x, y, z):
    return (x + y) / z


# 함수 `fun()`를 호출하려면 세 개의 매개 변수 `x`, `y`, `z`에 해당하는
# 세 개의 인자를 지정해야 한다.
# 아래 코드에서는 2, 3, 4를 각각 `x`, `y`, `z`의 인자로 지정하여 `fun()` 함수를 호출하고 
# 실행 후 반환된 값을 변수 `result`에 할당하였다.

# In[16]:


result = fun(2, 3, 4) # x=2, y=3, z=4


# 함수가 반환하는 값은 다음과 같다.

# In[17]:


print(result)


# 아래와 같이 함수의 반환값을 바로 확인할 수도 있다.

# In[18]:


fun(2, 3, 4)


# :::{admonition} 함수의 매개 변수와 인자
# :class: info
# 
# 매개 변수, 영어로 parameter(파라미터)는 함수를 선언할 때 사용되는 변수를 가리킨다.
# `fun()` 함수는 `x`, `y`, `z` 세 개의 매개 변수를 사용한다.
# 매개 변수는 함수를 호출할 때 지정되는 인자를 함수의 본문에 전달하는 역할을 수행한다.
# 참고로 인자는 영어로 argument(아규먼트)라 한다.
# 
# 매개 변수는 함수를 정의할 때 사용되는 특별한 변수이며, 
# 함수 내부에서만 의미를 갖기에 대표적인 **지역 변수**<font size='2'>local variable</font>이다. 
# 
# 프로그래밍에서 파라미터<font size='2'>parameter</font>는 다양한 의미로 사용된다. 
# 데이터 분석, 특히 머신러닝에서는 모델의 파라미터 개념이
# 매우 중요하다.
# :::

# **메서드 호출**

# 특정 객체의 메서드를 호출하는 방식도 기본적으로 동일하다.
# 다만, 객체와 메서드를 함께 이용한다.
# 예를 들어, 아래 코드는 리스트에 항목을 추가하는 `append()` 라는
# 리스트 메서드의 활용법을 보여준다.

# In[19]:


x = [1, 2, 3]
x.append(4)
print(x)


# **함수 인자 종류**

# 파이썬 함수의 인자는 두 종류로 나뉜다. 
# 
# * **위치 인자**<font size='2'>positional argments</font>: 인자를 지정된 위치에 사용해야 함.
# * **키워드 인자**<font size='2'>keyword arguments</font>: 키워드와 함께 지정되는 인자. 해당 인자의 기본값을 미리 지정한 후에
#     상황에 따라 다른 인자를 사용하는 것을 허용하고자 할 때 사용하는 인자.
#     위치 인자를 받는 매개 변수는 키워드 인자를 받는 키워드 매개 변수보다
#     먼저 선언되어야 한다.

# `print()` 함수를 이용하여 위치 인자와 키워드 인자를 살펴보자.
# `print()` 함수의 인자의 종류를 확인하기 위해 `help()` 함수를 이용한다.
# `help()` 함수는 지정된 값에 대한 정보를 알려준다.

# In[20]:


help(print)


# 위 정보에 의하면 `print()` 함수는 `sep=' '`, `end='\n'`, `file=sys.stdout`, `flush=False` 등 총 4개의
# 키워드 인자를 갖는다.
# 반면에 `value, ...` 로 표시된 부분은 최소 1 개 이상의 위치 인자가 사용되어야 함을 의미한다.

# 예를 들어, 아래 코드는 출력하려는 3 개의 값만 위치 인자로 지정하였다.

# In[21]:


print(1, 2, 3)


# `sep` 키워드 매개 변수는 2 개 이상의 값을 출력할 때 값들을 구분하는 방식을 지정한다.
# 기본값으로 지정한 `sep=' '`의 의미는 값들을 한 개의 스페이스로 구분한다는 의미이다.
# 예를 들어 두 개의 하이픈으로 값들을 분리하려면 `sep='--'`로 지정하면 된다.

# In[22]:


print(1, 2, 3, sep='--')


# `end` 키워드 매개 변수는 출력을 마친 후에 추가로 할 일을 지정한다.
# 기본값 `end='\n'`은 줄바꿈을 의미한다. 

# In[23]:


print(1, 2, 3, sep='--') # 줄바꿈 자동 실행
print(4, 5, 6, sep='--') # 줄바꿈 자동 실행


# 하지만 줄바꿈을 예를 들어 쉼표 스페이스로 대체하면 다음과 같이 작동한다.

# In[24]:


print(1, 2, 3, sep='--', end='--')
print(4, 5, 6, sep='--')


# :::{admonition} 구문 오류
# :class: warning
# 
# 반드시 위치 인자를 모두 지정한 다음에 키워드 인자를 지정해야 한다.
# 그렇지 않으면 구문 오류인 `SyntaxError`가 발생한다.
# 
# ```python
# In [16]: print(end='--', 1)
#   File "<ipython-input-1-8d02202c7b20>", line 1
#     print(end='--', 1)
#                      ^
# SyntaxError: positional argument follows keyword argument
# ```
# :::

# 반면에 키워드 인자들 사이의 순서는 중요하지 않다.

# In[25]:


print(1, 2, 3, end='--', sep='--')
print(4, 5, 6, sep='--')


# **키워드 인자를 사용하는 함수의 정의**

# 키워드 인자는 파이썬 프로그래밍언어의 주요 특징 중에 하나이다.
# 반면에 자바<font size='2'>Java</font>, C, C++ 등은 키워드 인자 기능을 지원하지 않는다.
# 파이썬 이외에 자바스크립트<font size='2'>JavaScript</font>, C# 등은 키워드 인자를 지원한다.
# 
# 앞으로 살펴보게 될 넘파이, 판다스 라이브러리에 포함된 대다수의 함수가 키워드 인자를 활용한다.
# 여기서는 키워드 인자를 사용하는 함수를 정의하는 방식을 간단한 예제를 통해 살펴본다.
# 
# 온라인 서점에서는 책을 정가의 10%를 할인해서 판매하는데 
# 특별한 날엔 정가의 20%를 할인해주는 행사를 진행한다고 가정하자.
# 이때, 일반적으로 10%의 할인을 적용하다가 행사가 진행되는 20%를 할인하는 함수를 다음과 같이 
# 정의할 수 있다.

# In[26]:


def book_price(price, discount=0.1):
    return price * (1 - discount)


# `book_price()` 함수의 정의에 사용되는 두 매개 변수<font size='2'>parameter</font>는
# `price`와 `discount`이다.
# `price`는 책의 정가를 위치 인자로 받는 매개 변수이다.
# 반면에 `discount`는 책값 할인율을 인자로 받는 매개 변수이며
# 10% 할인이 기본 키워드 인자로 지정되었다.

# 이제 정가 23,000원인 책을 구입할 때 실제로 지급해야 하는 값은 10% 할인된 20,700원이다.
# 
# - `book_price(23000)`: 위치 인자를 받는 `price` 매개 변수에 대한 인자 23000 만 지정한다.
#     반면에 할인율은 10% 기본값을 사용하기에 별도로 지정하지 않는다.

# In[27]:


discount = 0.1
print(f"책값은 {int(100*discount)}% 할인해서 {book_price(23000)}원입니다.")


# 이제, 예를 들어, 여름 특별 행사로 인해 20%를 할인해 준다고 가정하자.
# 그러면 정가 23,000원에 대해 지불해야 하는 값은 18,400원이다.
# 
# - `book_price(23000, discount=0.2)`: 할인율이 기본값인 10%가 아닌 20%를 적용해야 하기에 별도로 지정한다.

# In[28]:


print(f"책값은 {int(100*discount)}% 할인해서 {book_price(23000, discount=0.2)}원입니다.")


# ## 이항 연산자와 비교문

# 사칙연산과 비교문의 사용법은 일반적으로 알려진 방식과 동일하다.

# In[29]:


5 - 7


# In[30]:


12 + 21.5


# In[31]:


5 <= 2


# 부등호 연산자는 여러 개를 종합하여 사용할 수도 있다.

# In[32]:


4 > 3 >= 2 > 1


# 사칙연산과 비교문에 사용되는 연산자 모두 두 개의 값을 이용하여
# 새로운 값을 계산하는 이항 연산자이다. 
# 파이썬에서 지원하는 주요 이항 연산자는 다음과 같다.
# 
# | 이항 연산자 | 설명 |
# | ---: | :--- |
# | a + b | a와 b를 더한다|
# | a - b | a에서 b를 뺀다|
# | a * b | a와 b를 곱한다|
# | a / b | a를 b로 나눈다|
# | a // b | a를 b로 나눈 몫을 계산한다|
# | a ** b | a의 b승을 구한다|
# | a == b | a와 b가 동일한 값을 가리키는지 여부 판단|
# | a != b | a와 b가 서로 다른 값을 가리키는지 여부 판단|
# | a < b | a가 b보다 작은지 여부 판단|
# | a <= b | a가 b보다 작거나 같은지 여부 판단|
# | a > b | a가 b보다 큰지 여부 판단|
# | a >= b | a가 b보다 크거나 같은지 여부 판단|
# | a is b | a와 b가 동일한 위치에 저장된 값을 참조하는지 여부 판단|
# | a is not b | a와 b가 다른 위치에 저정된 값을 참조하는지 여부 판단|

# **`is` 대 `==`**

# 세 개의 변수 `a`, `b`, `c`를 아래처럼 선언하자.

# In[33]:


a = [1, 2, 3]
b = a
c = list(a)


# 앞서 참조에 대해서 설명한 것처럼 각 변수가 참조하는 리스트는 아래 그림에서와 같다.

# <img src="https://github.com/codingalzi/pydata/blob/master/notebooks/images/variables-a-b-3.png?raw=1" style="width:330px;">

# 먼저, `is` 연산자는 동일한 위치에 저정된 값을 참조하는지 여부를 결정한다.
# 예를 들어, `a`와 `b`는 동일한 리스트를 참조한다.

# In[34]:


a is b


# 반면에 `a`와 `c `는 서로 다른 리스트를 참조한다.
# 
# **참고:** `is not`은 서로 다른 위치에 저정된 값을 참조할 때 참이다.

# In[35]:


a is c


# In[36]:


a is not c


# 반면에, `==` 연산자는 두 변수가 참조하는 값이 동일한 값인지 여부를 판단한다.
# 예를 들어, 두 변수 `a`, `c`가 비록 서로 다른 위치에 저정된 값을 참조하기는 하지만
# 참조된 두 값 모두 리스트 `[1, 2, 3]` 으로 동일한 값이다.

# In[37]:


a == c


# 아래는 당연히 성립한다.

# In[38]:


a == b


# `!=` 는 `==`와 반대로 작동한다.

# In[39]:


a != c


# In[40]:


a != b


# :::{admonition} 기타 연산자
# :class: info
# 
# 앞서 언급된 산술 연산자 이외에 논리 연산자, 비트 연산자가 사용된다.
# 논리 연산자는 아래에서 좀 더 살펴볼 예정이지만 비트 연산자는 여기서는 다루지 않는다.
# 비트 연산에 대한 기초 지식은 
# [파이썬 코딩도장: 비트 연산자 사용하기](https://dojang.io/mod/page/view.php?id=2460)에서 확인할 수 있다.
# :::

# ## 객체와 자료형

# **객체**

# 파이썬이 다루는 대상(값)은 모두 **객체**<font size='2'>object</font>,
# 즉 특정 클래스의 인스턴스다.
# 따라서 값과 관련된 적절한 메서드<font size='2'>methods</font>를 활용할 수 있어야 한다.
# 앞으로 문자열, 리스트, 넘파이 어레이, 판다스 데이터프레임 등의 메서드를 적절하게
# 활용하는 방법을 살펴본다.

# **객체의 속성과 메서드**

# 모든 객체는 **속성**<font size='2'>attributes</font>과 **메서드**<font size='2'>methods</font>를 갖는다.
# 
# * 속성: 객체 안에 포함된 값 또는 해당 값에 대한 정보
# * 메서트: 객체 안에 포함된 값을 조작하는 함수

# 객체 이름 바로 뒤에 마침표를 작성한 후에 <kbd>Tab</kbd> 키를 이용하면 주어진 객체의 모든 속성과 메서드를 확인할 수 있다.

# ```python
# In [1]: a = 'foo'
# 
# In [2]: a.<Tab>
# a.capitalize  a.format      a.isupper     a.rindex      a.strip
# a.center      a.index       a.join        a.rjust       a.swapcase
# a.count       a.isalnum     a.ljust       a.rpartition  a.title
# a.decode      a.isalpha     a.lower       a.rsplit      a.translate
# a.encode      a.isdigit     a.lstrip      a.rstrip      a.upper
# a.endswith    a.islower     a.partition   a.split       a.zfill
# a.expandtabs  a.isspace     a.replace     a.splitlines
# a.find        a.istitle     a.rfind       a.startswith
# ```

# 속성과 메서드를 확인하기 위해 `getattr()` 함수를 활용할 수도 있다.
# 예를 들어, 문자열의 `split()` 메서드를 확인하면 아래의 결과를 보여준다.
# 즉, `split`이 문자열 자료형의 메서드임을 확인해준다.

# In[41]:


a = 'foo'

getattr(a, 'split')


# **동적 참조**

# 변수에 할당된 값은 심지어 다른 자료형의 값으로 변경될 수 있으며, 
# 그에 따른 자료형의 정보도 함께 변경되어 저장된다.
# 이렇게 작동하는 방식을 **동적 참조**<font size='2'>dynamic reference</font>라 한다.
# 
# `type()` 함수는 인자의 자료형(타입)을 확인해준다.

# In[42]:


a = 'hello'
type(a)


# In[43]:


a = ['h', 'e', 'l', 'l', 'o']
type(a)


# **객체의 자료형 정보 활용**

# 객체의 자료형에 따라 다른 일을 지정할 수도 있다.
# 이를 위해 먼저 객체의 자료형이 지정된 자료형인지 확인하는 기능이 필요하다.
# 예를 들어, `a`가 가리키는 값이 정수형의 값인지를 다음과 같이 확인한다.

# In[44]:


a = 5
isinstance(a, int)


# 부동소수점의 자료형인가를 확인하려면 다음과 같이 한다.

# In[45]:


b = 4.5
isinstance(b, float)


# 여러 자료형 중의 하나인가를 확인하려면 여러 자료형을 튜플로 작성하여 사용한다.
# 예를 들어, 정수 또는 부동소수점 중의 하나의 값인가를 확인하려면 다음과 같이 한다.

# In[46]:


isinstance(a, (int, float))


# In[47]:


isinstance(b, (int, float))


# 아래 코드는 정수 또는 부동소수점인지에 따라 다른 일을 수행하도록 하는 전형적인 예이다.

# In[48]:


def select_type(x):
    if isinstance(x, int):
        return x
    elif isinstance(x, float):
        return int(x)
    else:
        return "정수도 부동소수점도 아님!"


# In[49]:


select_type(3)


# In[50]:


select_type(3.14)


# In[51]:


select_type('3.14')


# **강타입 대 약타입**

# 파이썬의 모든 값은 특정 클래스의 객체라고 하였다.
# 따라서 사용되는 함수 또는 연산자에 따라 적절한 클래스의 객체를 사용하지 않으면
# 실행오류가 발생할 수 있다.
# 이처럼 사용되는 값의 자료형이 맞지 않을 경우 제대로 작동하지 않으면 **강타입**<font size='2'>strong types</font>라 부른다.
# 
# 예를 들어 문자열과 숫자는 더할 수 없다.

# ```python
# In [6]: '5' + 5
# ---------------------------------------------------------------------------
# TypeError                                 Traceback (most recent call last)
# <ipython-input-6-4dd8efb5fac1> in <module>
# ----> 1 '5' + 5
# 
# TypeError: can only concatenate str (not "int") to str
# ```

# 반면에, 예를 들어, 부동소수점과 정수의 나눗셈은 정수를 부동소수점으로 
# 강제로 형변환을 시켜서 실행된다.

# In[52]:


a = 4.5
b = 2

a / b


# 이렇게 작동하는 언어를 **약타입**<font size='2'>weak types</font> 언어라 부른다.
# 파이썬은 기본적으로 강타입 언어이지만, 약간의 약타입을 지원하는 언어라는 의미이다.

# ## 모듈

# **모듈**(module)은 파이썬 소스코드를 담고 있는, 확장자가 `.py`인 파이썬 스크립트 파일이다.

# **모듈 불러오기**

# 다음 내용을 담은 모듈 `some_module.py`가 현재 주피터 노트북이 실행되고 있는 디렉토리에
# 저장되어 있다고 가정하자.

# ```python
# PI = 3.14159
# 
# def f(x):
#     return x + 2
# 
# def g(a, b):
#     return a + b
# ```

# 아래 코드를 실행하려면 먼저 `some_module.py` 파일을 다운로드한다.

# In[53]:


import urllib.request
url = "https://raw.githubusercontent.com/codingalzi/datapy/master/jupyter-book/some_module.py"
urllib.request.urlretrieve(url, "./some_module.py")


# 모듈 `some_module`을 불러와서 그 안에 정의된 함수와 변수를 사용하려면
# 다음과 같이 모듈 이름과 함께 사용한다.
# 문자열에 포함된 `\t`는 탭 키를 한 번 누른 효과를 낸다.

# In[54]:


import some_module

result = some_module.f(5)
pi = some_module.PI

print(f"result:\t {result}")
print(f"pi:\t {pi}")


# 모듈에서 특정 변수와 함수만을 불러오면 모듈 이름을 사용할 필요가 없다.

# In[55]:


from some_module import g, PI

g(5, PI)


# **모듈 별칭**

# 모듈에 별칭을 주려면 `as` 예약어를 사용하여 지정한다.

# In[56]:


import some_module as sm

sm.f(pi)


# 함수 또는 변수만 따로 불러올 때에서 별칭을 줄 수 있다. 
# 역시 `as` 예약어를 이용한다.

# In[57]:


from some_module import PI as pi, g as gf

gf(6, pi) + pi


# 앞으로 많이 활용할 `numpy`, `pandas`, `matplotlib` 패키지의 `pyplot` 모듈은
# 관용적으로 사용되는 별칭이 있다.

# In[58]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ## 불변 객체 대 가변 객체

# **가변 객체**

# 리스트, 사전, 넘파이 어레이 등은 항복 변경, 추가, 삭제 등이 가능한<font size>mutable</font> 자료형이다.
# 
# 예를 들어, 리스트의 항목을 교체할 수 있다.

# In[59]:


a_list = ['foo', 2, [4, 5]]
a_list[2] = (3, 4)

a_list


# 리스트에 새로운 항목을 추가할 수도 있다.

# In[60]:


a_list.append('four')

a_list


# **불변 객체**

# 반면에 튜플은 항목 변경, 추가, 삭제 등이 절대로 불가능<font size>immutable</font>하다.

# ```python
# In [2]: a_tuple = (3, 5, (4, 5))
# 
# In [3]: a_tuple[1] = 'four'
# ---------------------------------------------------------------------------
# TypeError                                 Traceback (most recent call last)
# <ipython-input-3-23fe12da1ba6> in <module>
# ----> 1 a_tuple[1] = 'four'
# 
# TypeError: 'tuple' object does not support item assignment
# ```

# 문자열도 마찬가지로 어떤 변경도 허용되지 않는다.

# ```python
# In [4]: a_string = '123'
# 
# In [5]: a_string[2]='4'
# ---------------------------------------------------------------------------
# TypeError                                 Traceback (most recent call last)
# <ipython-input-5-0982fca81912> in <module>
# ----> 1 a_string[2]='4'
# 
# TypeError: 'str' object does not support item assignment
# ```
