#!/usr/bin/env python
# coding: utf-8

# # 기본 자료형

# **주요 내용**
# 
# * 파이썬 언어 기초 문법
# * 파이선 기본 자료형

# ## 기본 문법

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

# **세미콜론**

# 모든 명령문은 한 줄에 완성하는 것이 기본이며,
# 연속된 명령문은 줄바꿈을 해서 작성한다.
# 하지만 매우 간단한 명령문 여러 개를 한 줄에 연속으로 작성할 경우 세미콜론(`;`)을 사용할 수 있다.
# 
# 아래 코드는 세 개의 변수 할당을 한 줄에 처리한다.

# In[2]:


a = 5; b = 6; c = 7


# 하지만 세미콜론 사용 대신
# 아래와 같이 줄바꿈을 하여 각각의 명령문을 명확하게 작성하는 것을 추천한다.
# 그렇게 하면 보다 명료한 코드를 작성하게 된다.

# In[3]:


a = 5
b = 6
c = 7


# 연속된 변수 할당의 경우 파이썬은 아래 방식도 지원한다.

# In[4]:


a, b, c = 5, 6, 7
print(f"a: {a}", f"b: {b}", f"c: {c}", sep='\n')


# **주의사항:** `f"a: {a}, ..."`는 **f-문자열**, 즉 변수를 포함하는 문자열을
# 생성하는 **문자열 포매팅**<font size='2'>string formatting</font>이다.
# 문자열 포매팅에 대한 보다 자세한 설명은 [파이썬 프로그래밍 기초: 문자열 포매팅](https://codingalzi.github.io/pybook/datatypes.html?highlight=format#id10)를 참고한다.

# **객체**

# 파이썬이 다루는 대상(값)은 모두 **객체**<font size='2'>object</font>이다.
# 즉, 모든 값은 특정 클래스의 인스턴스이라는 의미이다.
# 따라서 값(객체)과 관련된 다양한 메서드<font size='2'>methods</font>가 존재하며 이런 메서드들을 
# 적절하게 활용하여 원하는 값을 생성하는 프로그램을 작성해야 한다.
# 
# 여기서는 앞으로 문자열<font size='2'>strings</font>, 
# 리스트<font size='2'>lists</font>, 넘파이 어레이<font size='2'>numpy arrays</font>, 
# 판다스 데이터프레임<font size='2'>pandas dataframes</font> 등의 다양한 메서드를 적절하게
# 활용하는 방법을 다룬다.

# **주석**

# 샵 기호(`#`) 다음에 오는 문장은 파이썬 인터프리터에의 의해 무시된다.
# 주로 코드의 일부 기능을 잠시 해제하거나 코드에 대한 설명을 전달하는 주석으로 활용된다.
# 예를 들어 아래 코드에서 주석은 코드 일부를 실행 해제하는 기능으로 사용되었다.

# In[5]:


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

# In[6]:


items = ["라디오", "", "tv", "전화"]
results = []

for item in items:
    if len(item) == 0:
        continue
    results.append(item)

print(results)


# 아래와 같이 명령문 끝 부분에 주석을 달아 
# 해당 명령문에 대한 설명 또는 정보를 제공하기도 한다.

# In[7]:


print("여기까지 공부했습니다.") # 여기까지 진행한 것을 확인해주는 문장 출력


# **함수 호출**

# 함수를 적절한 인자와 함께 실행하는 것을 **함수 호출**<font size='2'>function call</font>이라 한다.
# 예를 들어, 아래와 같이 세 개의 인자를 받는 함수를 정의한다.

# In[8]:


def fun(x, y, z):
    return (x + y) / z


# 함수 `fun()`를 호출하려면 세 개의 매개 변수 `x`, `y`, `z`에 해당하는
# 세 개의 인자를 지정해야 한다.
# 아래 코드에서는 2, 3, 4를 각각 `x`, `y`, `z`의 인자로 지정하여 `fun()` 함수를 호출하고 
# 실행 후 반환된 값을 변수 `result`에 할당하였다.

# In[9]:


result = fun(2, 3, 4) # x=2, y=3, z=4


# 함수가 반환하는 값은 다음과 같다.

# In[10]:


print(result)


# 아래와 같이 함수의 반환값을 바로 확인할 수도 있다.

# In[11]:


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

# In[12]:


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

# In[13]:


help(print)


# 위 정보에 의하면 `print()` 함수는 `sep=' '`, `end='\n'`, `file=sys.stdout`, `flush=False` 등 총 4개의
# 키워드 인자를 갖는다.
# 반면에 `value, ...` 로 표시된 부분은 최소 1 개 이상의 위치 인자가 사용되어야 함을 의미한다.

# 예를 들어, 아래 코드는 출력하려는 3 개의 값만 위치 인자로 지정하였다.

# In[14]:


print(1, 2, 3)


# `sep` 키워드 매개 변수는 2 개 이상의 값을 출력할 때 값들을 구분하는 방식을 지정한다.
# 기본값으로 지정한 `sep=' '`의 의미는 값들을 한 개의 스페이스로 구분한다는 의미이다.
# 예를 들어 두 개의 하이픈으로 값들을 분리하려면 `sep='--'`로 지정하면 된다.

# In[15]:


print(1, 2, 3, sep='--')


# `end` 키워드 매개 변수는 출력을 마친 후에 추가로 할 일을 지정한다.
# 기본값 `end='\n'`은 줄바꿈을 의미한다. 

# In[16]:


print(1, 2, 3, sep='--') # 줄바꿈 자동 실행
print(4, 5, 6, sep='--') # 줄바꿈 자동 실행


# 하지만 줄바꿈을 예를 들어 쉼표 스페이스로 대체하면 다음과 같이 작동한다.

# In[17]:


print(1, 2, 3, sep='--', end='--')
print(4, 5, 6, sep='--')


# :::{admonition} 문법 오류
# :class: warning
# 
# 반드시 위치 인자를 모두 지정한 다음에 키워드 인자를 지정해야 한다.
# 그렇지 않으면 문법 오류인 `SyntaxError`가 발생한다.
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

# In[18]:


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

# In[19]:


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

# In[20]:


discount = 0.1
print(f"책값은 {int(100*discount)}% 할인해서 {book_price(23000)}원입니다.")


# 이제, 예를 들어, 여름 특별 행사로 인해 20%를 할인해 준다고 가정하자.
# 그러면 정가 23,000원에 대해 지불해야 하는 값은 18,400원이다.
# 
# - `book_price(23000, discount=0.2)`: 할인율이 기본값인 10%가 아닌 20%를 적용해야 하기에 별도로 지정한다.

# In[21]:


print(f"책값은 {int(100*discount)}% 할인해서 {book_price(23000, discount=0.2)}원입니다.")


# **참조**

# 변수가 가리키는 값이 리스트와 같이 좀 복잡한 객체일 때는 파이썬 실행기 내부에서
# 변수가 해당 값을 **참조**(reference)한다.
# 참조는 변수가 단순히 하나의 값을 가리키는 것 이외에 부가적인 기능도 수행한다.
# 
# 예를 들어 변수 `a`가 리스트 `[1, 2, 3]`을 참조하도록 하자.

# In[22]:


a = [1, 2, 3]


# 그리고 변수 `b`를 다음과 같이 선언하면 변수 `a`와 동일한 값을 참조한다. 

# In[23]:


b = a


# 아래 그림에서에서처럼 `a`와 `b`가 화살표로 동일한 리스트를 가리키는 방식으로
# 참조를 표현할 수 있다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/datapy/master/jupyter-book/images/variables-a-b-1.png" style="width:340px;"></div>

# `a`와 `b`가 동일한 값을 참조하기에 `a`가 참조하는 값을 변화시키면 `b`도 영향을 받는다.
# 아래 코드는 `a`가 가리키는 리스트에 항목을 추가하면 `b`가 동일한 리스트를 가리키기에
# `b`가 가리키는 값도 함께 변하는 것을 보여준다.

# In[24]:


a.append(4) # 항목 추가
b


# 그림으로 표현하면 다음과 같다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/datapy/master/jupyter-book/images/variables-a-b-2.png" style="width:340px;"></div>

# 반면에 정수와 같이 보다 간단한 객체를 변수에 할당하는 경우는 다르게 작동한다.

# In[25]:


a = 4
b = a


# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/datapy/master/jupyter-book/images/variables-a-b-4.png" style="width:360px;"></div>

# 변수 `a`가 가리키는 값을 변경하더라도 변수 `b`가 가리키는 값은 영향을 받지 않는다.

# In[26]:


a = a + 1

print(f"a = {a}", f"b = {b}", sep="\n")


# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/datapy/master/jupyter-book/images/variables-a-b-5.png" style="width:360px;"></div>

# ### 전역 변수와 지역 변수

# 함수 밖에서 선언된 **전역 변수**(global variables)는 함수 내에서 사용할 수 있지만,
# 함수의 매개 변수 또는 함수 본문 내에서 선언된 **지역 변수**(local variables)는 함수 밖에서 사용할 수 없다.

# In[27]:


def append_element(some_list, element):
    some_list.append(element)


# 아래 코드에서 `data`는 함수 밖에서 선언된 전역 변수이기에
# 함수에 인자로 전달될 수 있다.

# In[28]:


data = [1, 2, 3]

append_element(data, 4)


# In[29]:


data


# 반면에 매개 변수로 지정된 `element`는 함수 실행이 멈춘 후에는 더 이상 사용할 수 없다.

# In[30]:


print(element)


# ### 동적 참조(dynamic reference)

# 변수에 할당된 값은 심지어 다른 자료형의 값으로 변경될 수 있으며, 
# 그에 따른 자료형의 정보도 함께 변경되어 저장된다.
# 이렇게 작동하는 방식을 **동적 참조**라 한다.

# In[ ]:


a = 5

type(a)


# In[ ]:


a = 'foo'

type(a)


# ### 강타입 대 약타입

# 파이썬의 모든 값은 특정 클래스의 객체라고 하였다.
# 따라서 사용되는 함수 또는 연산자에 따라 적절한 클래스의 객체를 사용하지 않으면
# 실행오류가 발생할 수 있다.
# 이처럼 사용되는 값의 자료형이 맞지 않을 경우 제대로 작동하지 않으면 **강타입(strong types)**라 부른다.
# 
# 예를 들어 문자열과 숫자는 더할 수 없다.

# In[31]:


'5' + 5


# 반면에, 예를 들어, 부동소수점과 정수의 나눗셈은 정수를 부동소수점으로 
# 강제로 형변환을 시켜서 실행된다.

# In[32]:


a = 4.5
b = 2

a / b


# 이렇게 작동하는 언어를 **약타입(weak types)** 언어라 부른다.
# 파이썬은 기본적으로 강타입 언어이지만, 약간의 약타입을 지원하는 언어라는 의미이다.

# ### 객체의 자료형 활용

# 객체의 자료형에 따라 다른 일을 지정할 수도 있다.
# 이를 위해 먼저 객체의 자료형이 지정된 자료형인지 확인하는 기능이 필요하다.
# 
# 예를 들어, `a`가 가리키는 값이 정수형의 값인지를 다음과 같이 확인한다.

# In[33]:


a = 5

isinstance(a, int)


# 그리고 부동소수점의 자료형인가를 확인하려면 다음과 같이 한다.

# In[34]:


b = 4.5

isinstance(b, (int, float))


# 여러 자료형 중의 하나인가를 확인하려면 여러 자료형을 튜플로 작성하여 사용한다.
# 예를 들어, 정수 또는 부동소수점 중의 하나의 값인가를 확인하려면 다음과 같이 한다.

# In[35]:


isinstance(a, (int, float))


# In[36]:


isinstance(b, (int, float))


# ### 객체의 속성과 메서드

# 모든 객체는 **속성**(attributes)과 **메서드**(methods)를 갖는다.
# 
# * 속성: 객체와 관련된 정보
# * 메서트: 객체의 속성을 조작하는 기능을 가진 함수

# 탭 키(<kbd>Tab</kbd>)를 이용하면 주어진 객체의 모든 속성과 메서드를 확인할 수 있다.

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

# In[37]:


a = 'foo'

getattr(a, 'split')


# 즉, `split` 은 함수(function), 즉, 메서드임을 확인해준다.

# ### 덕 타이핑(Duck typing)

# **덕 타이핑**은 "특정 기능을 지원하는가만 중요하다"는 의미를 전달할 때 사용하는 표현이다. 
# ("오리처럼 꽥꽥 울기만 하면 그것은 오리다" 라는 의미에서 만들어진 표현임)
# 
# 예를 들어, 문자열, 튜플, 리스트 등 처럼 각 항목을 차례대로 순환할 수 있는 값은
# `**iter**()` 메서드를 가지며, 이런 객체를 **이터러블**(iterable) 객체라 부른다.
# 즉, 어떤 객체이든 `**iter**()` 메서드만 지원하면 순환기능을 사용할 수 있으며,
# 그런 객체를 이터러블이라고 부른다.

# In[38]:


'123'.**iter**()


# In[39]:


(1, 2, 3).**iter**()


# In[40]:


[1, 2, 3].**iter**()


# 아래 함수는 이터러블 객체인지 여부를 판단해준다.
# 
# **참고:** `iter()` 함수는 인자가 `**iter()**` 메서드를 갖고 있다면 그 메서드를 호출하고,
# 아님면 오류를 발생시킨다.

# In[41]:


def isiterable(obj):
    try:
        iter(obj)
        return True
    except TypeError: # 이터러블하지 않음
        return False


# In[42]:


isiterable('a string')


# In[43]:


isiterable([1, 2, 3])


# 정수는 이터러블하지 않다.

# In[44]:


isiterable(5)


# `isiterable()` 함수를 이용하여
# 리스트는 아니지만 이터러블한 값을 모두 리스트로 형변환 시켜주는 함수를
# 아래와 같이 구현할 수 있다.

# In[45]:


def toList(x):
    if not isinstance(x, list) and isiterable(x):
        return list(x)


# 이제 문자열과 튜플을 리스트로 변환할 수 있다.

# In[46]:


toList("123")


# In[47]:


toList((1,2,3))


# **참고:** `toList()` 함수는 사실 `list()` 함수와 동일한 기능을 수행한다.

# In[48]:


list("123")


# In[49]:


list((1,2,3))


# ### 모듈 불러오기

# **모듈**(module)은 파이썬 소스코드를 담고 있는, 확장자가 `.py`인 파이썬 스크립트 파일이다.
# 
# 다음 내용을 담은 모듈 `some_module.py`가 현재 주피터 노트북이 실행되고 있는 디렉토리에
# 저장되어 있다고 가정하자.

# ```python
# # some_module.py
# PI = 3.14159
# 
# def f(x):
#     return x + 2
# 
# def g(a, b):
#     return a + b
# ```

# **주의사항:** 구글 코랩에서 아래 코드를 실행하려면 먼저 `some_module.py` 파일을 구글 코랩에 업로드 해야 한다. 아니면 아래 코드를 실행해서 해당 코드를 다운로드할 수도 있다. 

# In[50]:


import urllib.request
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/codingalzi/pydata/master/"
url = DOWNLOAD_ROOT + "notebooks/some_module.py"
urllib.request.urlretrieve(url, "./some_module.py")


# 모듈 `some_module`을 불러와서(import) 그 안에 정의된 함수와 변수를 사용하는 방법은
# 다음과 같이 모듈이름과 함께 사용한다.

# In[51]:


import some_module


# In[52]:


result = some_module.f(5)
pi = some_module.PI

print(f"result:\t {result}", f"pi:\t {pi}", sep='\n')


# 모듈에서 특정 변수와 함수만을 불러오면 모듈 이름을 사용할 필요가 없다.

# In[53]:


from some_module import g, PI

g(5, PI)


# 모듈에 별칭을 주려면 `as` 예약어를 사용하여 지정한다.

# In[54]:


import some_module as sm

sm.f(pi)


# 함수 또는 변수만 따로 불러올 때에서 별칭을 줄 수 있다. 
# 역시 `as` 예약어를 이용한다.

# In[55]:


from some_module import PI as pi, g as gf

gf(6, pi) + pi


# ### 이항 연산자와 비교문

# 사칙연산과 비교문의 사용법은 일반적으로 알려진 방식과 동일하다.

# In[56]:


5 - 7


# In[57]:


12 + 21.5


# In[58]:


5 <= 2


# 부등호 연산자는 여러 개를 종합하여 사용할 수도 있다.

# In[59]:


4 > 3 >= 2 > 1


# 파이썬에서 지원하는 주요 이항 연산자는 다음과 같다.
# 
# | 이항 연산자 | 설명 |
# | :--- | :--- |
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

# #### `is`와 `==`의 차이

# 세 개의 변수 `a`, `b`, `c`를 아래처럼 선언하자.

# In[60]:


a = [1, 2, 3]
b = a
c = list(a)


# 앞서 참조에 대해서 설명한 것처럼 각 변수가 참조하는 리스트는 아래 그림에서와 같다.

# <img src="https://github.com/codingalzi/pydata/blob/master/notebooks/images/variables-a-b-3.png?raw=1" style="width:330px;">

# 먼저, `is` 연산자는 동일한 위치에 저정된 값을 참조하는지 여부를 결정한다.
# 예를 들어, `a`와 `b`는 동일한 리스트를 참조한다.

# In[61]:


a is b


# 반면에 `a`와 `c `는 서로 다른 리스트를 참조한다.
# 
# **참고:** `is not`은 서로 다른 위치에 저정된 값을 참조할 때 참이다.

# In[62]:


a is c


# In[63]:


a is not c


# 반면에, `==` 연산자는 두 변수가 참조하는 값이 동일한 값인지 여부를 판단한다.
# 예를 들어, 두 변수 `a`, `c`가 비록 서로 다른 위치에 저정된 값을 참조하기는 하지만
# 참조된 두 값 모두 리스트 `[1, 2, 3]` 으로 동일한 값이다.

# In[64]:


a == c


# 아래는 당연히 성립한다.

# In[65]:


a == b


# `!=` 는 `==`와 반대로 작동한다.

# In[66]:


a != c


# In[67]:


a != b


# **참고:** 앞서 언급된 산술 연산자 이외에 논리 연산자, 비트 연산자가 사용된다.
# 논리 연산자는 아래에서 좀 더 살펴볼 예정이지만 비트 연산자는 여기서는 다루지 않는 
# 대신에 [파이썬 코딩도장: 비트 연산자 사용하기](https://dojang.io/mod/page/view.php?id=2460)를
# 추천한다.

# ### 변경 가능한(mutable) 객체와 변경 불가능한(immutable) 객체

# 리스트, 사전, 넘파이 어레이 등은 변경이 가능한 자료형이다.
# 
# 예를 들어, 리스트의 항목을 교체할 수 있다.

# In[68]:


a_list = ['foo', 2, [4, 5]]
a_list[2] = (3, 4)

a_list


# 리스트에 새로운 항목을 추가할 수도 있다.

# In[69]:


a_list.append('four')

a_list


# 반면에 문자열과 튜플은 항목 수정이 불가능하다.

# In[70]:


a_tuple = (3, 5, (4, 5))

a_tuple[1] = 'four'


# In[71]:


a_string = "123"
a_string[2]='4'

