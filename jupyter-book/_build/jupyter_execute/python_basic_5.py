#!/usr/bin/env python
# coding: utf-8

# (sec:python_basic_5)=
# # 모음 자료형

# {ref}`sec:python_basic_3`에서 단일값 객체의 자료형인 스칼라 자료형을 살펴보았다.
# 여기서는 여러 개의 값으로 이루어진 객체의 자료형을 살펴본다.
# 
# 여러 개의 값을 항목으로 갖는 객체는 항목들을 다루는 방식에 따라 구분되며,
# 그런 객체의 자료형을 통틀어 모음 자료형<font size='2'>collection</font>이라 한다.
# 파이썬이 기본으로 제공하는 모음 자료형은 다음과 같다.
# 
# * 튜플(`tuple`)
# * 리스트(`list`)
# * 사전(`dict`)
# * 집합(`set`)
# 
# 이 중에 튜플과 리스트는 항목의 순서가 중요하다는 의미에서 **순차 자료형**<font size='2'>sequence</font>이라 불리기도 한다.

# :::{admonition} 문자열과 모음 자료형
# :class: info
# 
# 문자열(`str`)도 모음 자료형처럼 취급한다. 실제로 앞으로 살펴볼 
# 인덱스, 인덱싱, 슬라이싱 등 순차 자료형과 관련된 기능을 문자열 또한 지원하기에 
# 문자열을 순차 자료형으로 간주하기도 한다.
# :::

# ## 튜플

# 튜플(tuple)은 여러 개의 값을 항목으로 가지며 소괄호로 감싼다.

# In[1]:


tup = (4, 5, 6)
tup


# 굳이 소괄호를 사용하지 않아도 되지만 권장되지 않는다.

# In[2]:


tup = 4, 5, 6
tup


# 하나의 항목을 괄호를 감싼다 해도 튜플로 간주되지 않음에 주의한다.

# In[3]:


singleton1 = (3)
singleton1


# In[4]:


singleton2 = ("abc")
type(singleton2)


# 한 개의 항목으로 이루어진 튜플을 생성하려면 쉼표를 추가해야 한다.

# In[5]:


tup3 = ("abc",)
tup3


# In[6]:


type(tup3)


# 하지만 하나의 항목을 갖는 튜플은 굳이 사용할 이유가 별로 없다.
# 이유는 튜플은 불변 자료형이기에 항목을 추가해서 튜플을 수정하는 일이 허영되지 않기 때문이다.

# **`tuple()` 형 변환 함수**

# `tuple()` 함수는 다른 모음 자료형을 튜플로 변환한다.

# In[7]:


tuple([4, 0, 2])


# In[8]:


tup= tuple('string')
tup


# **중첩 튜플**

# 튜플의 항목은 임의의 파이썬 객체가 사용될 수 있다.
# 즉, 튜플의 항목으로 튜플이 사용될 수 있다.

# In[9]:


nested_tup = (4, 5, 6), (7, 8)
nested_tup


# 물론 항목으로 문자열, 리스트, 사전 등 임의의 값이 사용될 수 있다. 

# In[10]:


nested_tup2 = (3, (4, 5, 6), [1, 2], "파이썬")
nested_tup2


# **인덱스와 인덱싱**

# 튜플 맨 왼편에 위치한 항목부터 차례대로  0, 1, 2, ... 로 시작하는 **인덱스**<font size='2'>index</font>를 갖는다.
# 인덱스를 이용하여 해당 위치의 항목을 확인할 수 있으며, 이를 **인덱싱**<font size='2'>indexing</font>이라 한다.
# 
# 예를 들어 `tup` 변수가 가리키는 튜플의 첫째 항목은 `s`라는 사실을
# 아래와 같이 인덱싱으로 확인할 수 있다.

# In[11]:


tup[0]


# 인덱스가 0부터 시작하기에 예를 들어 전체 항목의 수가 3이면 마지막 항목의 인덱스는 2이다.

# In[12]:


tup = ('foo', [1, 2], True)
tup[2]


# 튜플은 불변 자료형이기에 위 튜플의 마지막 항목을 인덱싱을 이용하여 `False`로 대체하고자 시도하면 오류가 발생한다.

# ```python
# In [3]: tup[2] = False
# ---------------------------------------------------------------------------
# TypeError                                 Traceback (most recent call last)
# Input In [3], in <cell line: 1>()
# ----> 1 tup[2] = False
# 
# TypeError: 'tuple' object does not support item assignment
# ```

# 튜플이 변경이 불가능한 자료형이라고 해서 튜플의 모든 항목이 모두 변경이 불가능학 객체이어야 하는 것은 아니다.
# 예를 들어 `tup`의 둘째 항목은 리스트 `[1, 2]` 인데, 리스트는 변경이 가능한(mutable) 자료형이다.
# 따라서 아래와 같이 둘째 항목 자체는 변경이 가능하다.

# In[13]:


tup[1].append(3)

tup


# 이런 성질이 가능한 이유는 다음과 같으며, 아래 그림과 함께 설명을 보다 잘 이해할 수 있다.
# 
# `tup`은 `('foo', [1, 2], True)`를 참조한다.
# 그리고 둘째 항목인 `[1, 2]` 또한 참조 형태로 다른 메모리에 저장된다.
# 즉 `tup`의 둘째 항목은 `[1, 2]`가 저장된 위치의 주소이다.
# 그런데 `[1, 2]`가 변경되어도 주소 자체는 변하지 않는다.
# 따라서 `tup` 입장에서는 변한 게 하나도 없게 된다.
# 참고로, 리스트의 주소는 첫째 항목이 저장된 위치의 주소를 사용한다.

# <변경 전>
# 
# <div align="center"><img src="https://github.com/codingalzi/pydata/blob/master/notebooks/images/tuple10.png?raw=1" style="width:400px;"></div>

# <변경 후>
# 
# <div align="center"><img src="https://github.com/codingalzi/pydata/blob/master/notebooks/images/tuple11.png?raw=1" style="width:400px;"></div>

# **튜플 이어붙이기: `+` 연산자**

# 두 개의 튜플을 이어붙인다. 

# In[14]:


(4, None, 'foo') + (6, 0)


# 튜플 여러 개를 이어붙일 수도 있다.

# In[15]:


(4, None, 'foo') + (6, 0) + ('bar',)


# **튜플 복제 후 이어붙이기: `*` 연산자**

# 지정된 정수만큼 튜플을 복사해서 이어붙인다.

# In[16]:


('foo', 'bar') * 4


# **튜플 해체**

# 튜플 항목 각각에 대해 변수를 지정하고자 할 때 튜플을 해체하는 기법을 사용한다.
# 단, 사용되는 변수의 수는 항목의 수와 일치해야 한다.
# 예를 들어, 세 개의 항목을 갖는 항목을 해체하려면 세 개의 변수가 필요하다.

# In[17]:


tup = (4, 5, 6)
a, b, c = tup


# 변수 `b`와 `c`는 각각 둘째, 셋재 항목을 가리킨다.

# In[18]:


b + c


# 굳이 이름을 주지 않아도 되는 항목이 있다면
# 변수 대신에 밑줄(underscore) 기호 `_`를 사용한다.
# 예를 들어 변수 `a`가 필요없다면 아래와 같이 튜플 해체를 해도 된다.

# In[19]:


tup = (4, 5, 6)
_, b, c = tup


# In[20]:


b + c


# 하지만 밑줄을 빼면 오류가 발생한다.

# ```python
# In [1]: tup = (4, 5, 6)
# 
# In [2]: b, c = tup
# ---------------------------------------------------------------------------
# ValueError                                Traceback (most recent call last)
# Input In [2], in <cell line: 1>()
# ----> 1 b, c = tup
# 
# ValueError: too many values to unpack (expected 2)
# ```

# 반면에 앞에 몇 개만 이름을 지정하고 나머지는 하나의 리스트로 묶을 수 있다.
# 이를 위해 별표 기호(asterisk) `*`를 하나의 변수이름과 함께 사용한다.

# In[21]:


values = (1, 2, 3, 4, 5)
a, b, *rest = values


# In[22]:


a


# In[23]:


b


# In[24]:


rest


# 나머지 항목들을 무시하고 싶다면 별표와 밑줄을 함께 사용한다.

# In[25]:


a, b, *_ = values


# In[26]:


print(a, b, sep=', ')


# 중첩 튜플을 풀어헤칠 때는 중첩 모양을 본딸 수도 있다.
# 예를 들어, 아래와 같이 하면 `c`는 셋째 항목의 첫째 항목을 가리킨다.

# In[27]:


tup = (4, 5, (6, 7))
a, b, (c, d) = tup
c


# 하지만 아래와 같이 하면 `c`는 튜플의 셋째 항목을 가리킨다.

# In[28]:


tup = (4, 5, (6, 7))
a, b, c = tup
c


# 튜플 해체를 이용하면, 여러 변수가 가리키는 값을 쉽게 바꿀 수 있다.
# 예를 들어, 변수 `a`, `b`가 각각 1과 2를 가리키도록 하자.

# In[29]:


a, b = 1, 2

print(f"a={a}, b={b}")


# 이제 아래와 같이 하면 `a`, `b`가 가리키는 값을 서로 바꾸게 된다.

# In[30]:


b, a = a, b

print(f"a={a}, b={b}")


# 튜플 해체는 `for` 반복문에서 유용하게 사용된다.
# 즉, 리스트의 항목이 일정한 크기의 튜플일 때 각각의 항목에 변수를 지정하여 활용한다.

# In[31]:


seq = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]

for a, b, c in seq:
    print('a={0}, b={1}, c={2}'.format(a, b, c))


# **튜플 메서드**

# 튜플은 변경이 불가능한 자료형이기에 제고되는 튜플 메서드가 많지 않으며,
# 특정 값이 항목으로 몇 번 사용되었가를 세어주는 `count()` 메서드와
# 특정 항목의 인덱스를 찾아주는 `index()` 메서드가 주로 사용된다.
# 
# 예를 들어, 아래 리스트에서 숫자 2는 4번 사용되었다.

# In[32]:


a = (1, 2, 2, 2, 3, 4, 2)
a.count(2)


# 반면에 2가 가장 먼저 사용된 위치의 인덱스는 1이다.

# In[33]:


a = (1, 2, 2, 2, 3, 4, 2)
a.index(2)


# ## 리스트

# 리스트 사용법은 튜플과 유사하다. 

# In[34]:


a_list = [2, 3, 7, None]


# `list()` 함수는 튜플, 문자열 등의 모음 자료형을 리스트로 변환한다.

# In[35]:


tup = ('foo', 'bar', 'baz')
b_list = list(tup)


# In[36]:


b_list


# **리스트 항목 변경, 추가, 삭제**

# 튜플과는 달리 리스트에 항목을 추가하거나, 특정 항목을 다른 항목으로 변경할 수 있으며,
# 리스트를 다루는 많은 메서드를 지원한다.

# In[37]:


b_list[1] = 'peekaboo'
b_list


# `append()` 메서드는 새로운 항목을 가장 오른편에 추가한다.

# In[38]:


b_list.append('dwarf')
b_list


# `insert()` 메서드는 지정된 인덱스 위치에 새로운 항목을 추가한다.

# In[39]:


b_list.insert(1, 'red')
b_list


# `pop()` 메서드는 지정된 인덱스 위치의 항목을 삭제한다.
# 그런데 단순히 삭제만 하는 것이 아니라 삭제되는 값을 반환한다.

# In[40]:


b_list.pop(2)


# In[41]:


b_list


# 인자를 지정하지 않으면 마지막 항목을 삭제한다.

# In[42]:


b_list.pop()


# In[43]:


b_list


# `remove()` 메서드는 지정된 항목을 삭제한다.
# 지정된 항목이 여러 번 사용되었을 경우 가장 작은 인덱스의 값을 삭제한다. 

# In[44]:


# 먼저 `foo`를 추가하여 중복 사용되게 만든다.
b_list.insert(1, 'foo')

b_list.remove('foo')
b_list


# `in()` 연산자는 특정 항목이 리스트에 포함되어 있는지 여부를 판단해준다. 

# In[45]:


'baz' in b_list


# In[46]:


'dwarf' in b_list


# In[47]:


'dwarf' not in b_list


# **리스트 이어붙이기; `+` 연산자**

# 두 개의 리스트를 이어붙여서 새로운 리스틀 생성한다.

# In[48]:


[4, None, 'foo'] + [7, 8, (2, 3)]


# **`extend()` 메서드**

# 주어진 리스트에 다른 지정된 리스트를 이어붙이는 방식으로 항목을 추가한다.
# 원래의 리스트를 수정하는 메서드이고, 따라서 반환값이 `None`임에 주의해야 한다.
# 항상 새로운 리스트를 생성하는 `+` 연산자보다 좀 더 빠르게 작동하며,
# 따라서 매우 긴 리스트를 이어붙일 때 기본적으로 선호된다.

# In[49]:


x = [4, None, 'foo']
x.extend([7, 8, (2, 3)])


# In[50]:


x


# **`sort()` 메서드**

# `sort()` 메서드는 항목을 크기 순으로 정렬한다.
# `sort()` 메서드의 반환값은 `None`이다. 
# 즉, 주어진 리스트의 항목을 크기 순으로 정렬하여 변경하지만 함수 자체의 반환값은 없다.

# In[51]:


a = [7, 2, 5, 1, 3]
a.sort()


# In[52]:


a


# 정렬할 때 사용되는 크기의 기준을 지정할 수 있다.
# 예를 들어, 문자열들을 기본값인 사전식 순서가 아니라 문자열들의 길이 기준으로 정렬하려면
# 항목의 크기를 계산하는 함수를 인자로 갖는 `key` 키워드의 인자를 `len()` 함수의 
# 이름인 `len`으로 지정하면 된다.

# In[53]:


b = ['saw', 'small', 'He', 'foxes', 'six']
b.sort(key=len)
b


# 참고로 `key` 키워드 인자를 지정하지 않은 알파벳 순서를 기준으로 삼는 사전식 순서로 정렬된다.
# 사전식 준서에서 영어 알파벳 대문자가 소문자보다 작은 것으로 간주된다.

# In[54]:


b.sort()
b


# **리스트 슬라이싱**

# 슬라이싱 용법은 문자열, 튜플 등의 경우와 동일하다. 
# 아래 코드는 1번부터 4번 인덱스의 값으로 이루어진 리스트를 생성한다.

# In[55]:


seq = [7, 2, 3, 7, 5, 6, 0, 1]
sub_seq = seq[1:5]

sub_seq


# <div align="center"><img src="https://github.com/codingalzi/pydata/blob/master/notebooks/images/list_slicing10.png?raw=1" style="width:420px;"></div>

# 위 그림에서 볼 수 있듯이 슬라이싱은 기존에 주어진 리스트를 수정하지 않으면서
# 구간 정보를 활용하여 새로운 리스트를 생성한다.

# In[56]:


seq


# 슬라이싱 기능을 이용하여 특정 위치부터 시작하는 구간에 여러 개의 항목을 추가할 수도 있다.
# 아래 코드는 3번, 4번 인덱스 위치의 값 대신에 4개의 원소를 추가 입력하는 것이다. 
# 기존에 5번 이상의 인덱스에 위치한 값들은 더 추가되는 값들의 수 만큰 오른편으로 
# 밀림에 주의한다.

# In[57]:


seq[3:5] = [6, 3, 8, 4]
seq


# <변경 전>
# 
# <div align="center"><img src="https://github.com/codingalzi/pydata/blob/master/notebooks/images/list_slicing11.png?raw=1" style="width:450px;"></div>

# <변경 후>
# 
# <div align="center"><img src="https://github.com/codingalzi/pydata/blob/master/notebooks/images/list_slicing12.png?raw=1" style="width:550px;"></div>

# 슬라이싱 구간의 시작과 끝을 지정하는 값을 필요에 따라 선택적으로 생략할 수도 있다.
# 생략된 값은 각각 리스트의 처음과 끝을 가리키는 값으로 처리된다.
# 아래 코드는 0번 인덱스부터 4번 인덱스까지의 구간을 대상으로 한다.

# In[58]:


seq[:5]


# 아래 코드는 3번 인덱스부터 리스트 오른편 끝가지를 대상으로 한다.

# In[59]:


seq[3:]


# 음수 인덱스는 리스트 오른편 부터 -1, -2, -3, 등으로 왼편으로 이동하면서 지정된다.
# 아래 코드는 끝에서 4번째부터 마지막까지 구간을 대상으로 한다.

# In[60]:


seq[-4:]


# 아래 코드는 끝에서 6번째부터 끝에서 두번째 이전, 즉, 끝에서 세번째까지 슬라이싱한다.

# In[61]:


seq[-6:-2]


# 구간의 처음과 끝이 모두 생략되면 리스트 전체를 대상으로 한다.
# 아래 코드는 리스트 전체를 대상으로 하지만 2 스텝씩 건너 뛰며 항목을 슬라이싱한다.
# 즉, 0, 2, 4, ... 등의 인덱스를 대상으로 한다.

# In[62]:


seq[::2]


# 음수의 스텝이 사용되면 역순으로 슬라이싱된다.
# 아래 코드는 리스트의 오른편 끝에서 왼편으로 역순으로 슬라이싱한다.
# 즉, 기존의 리스트의 항목을 뒤집어서 새로운 리스트를 생성한다.

# In[63]:


seq[::-1]


# 리스트의 마지막 항목의 인덱스는 아래 두 가지 방식으로 표현한다.
# 
# * 방법 1: 리스트의 길이에서 1을 뺀 값
# * 방법 2: -1
# 
# 따라서 위 코드는 아래 코드와 동일하다.

# In[64]:


seq[-1::-1]


# 아래 코드도 같다. 이유는 리스트의 길이가 10이기 때문이다.

# In[65]:


seq[9::-1]


# ## `range()` 함수

# 규칙성을 가진 정수들의 모음을 반환한다.
# 반환된 값은 이터러블 객체이며, 
# 리스트와 유사하게 작동한다.
# 예를 들어, 0부터 9까지의 정수들로 이루어진 `range` 객체는 다음과 같이 생성한다.

# In[66]:


range(10)


# :::{admonition} 이터러블 객체
# :class: info
# 
# 이터러블 객체에 대한 자세한 설명은 
# [이터러블, 이터레이터, 제너레이터](https://codingalzi.github.io/pybook/casestudy_collections.html)를 참고하라.
# :::

# `range(10)`은 `range(0, 10)`과 동일하다.
# 이때 첫째 인자 0은 구간의 시작을, 둘째 인자는 10은 구간의 끝보다 하나 큰 값을 가리킨다.
# 반환된 값의 자료형은 `range` 이다.

# In[67]:


type(range(10))


# `range(0, 10)` 안에 포함된 항목을 `for` 반복문을 이용하여 확인할 수 있다.

# In[68]:


for item in range(0, 10):
    print(item)


# 리스트로 형변환을 하면 보다 명확하게 확인된다.

# In[69]:


list(range(10))


# 슬라이싱에서 처럼 스텝을 사용할 수 있다.
# 예를 들어, 0에서 19까지의 정수중에서 짝수만으로 이루어진 `range` 객체는 다음과 같이 
# 스텝(step) 크기 2를 셋째 인자로 지정하여 생성한다.

# In[70]:


list(range(0, 20, 2))


# 스텝 크기를 음수로 지정하면 크기 역순으로 이루어진 `range` 객체를 생성한다.
# 
# __주의사항:__ 음수 스텝을 사용할 경우 둘째 인자는 원하는 구간보다 1이 작은 값을 사용해야 한다.

# In[71]:


list(range(5, 0, -1))


# In[72]:


list(range(5, 0, -2))


# **__`range()` 함수 주요 활용법 1__**

# 리스트 또는 튜플의 길이 정보를 이용하여 인덱싱을 활용하는 방식이 많이 사용된다.

# In[73]:


seq = [1, 2, 3, 4]
for i in range(len(seq)):
    val = seq[i]


# **__`range()` 함수 주요 활용법 2__**

# 매우 많은 항목을 담은 리스트 대신에 `range` 객체를 `for` 반복문과 함께 사용한다.
# 이유는 `range` 객체가 리스트보다 훨씬 적은 메모리를 사용하기 때문이다.
# (이에 대한 근거는 여기서는 다루지 않는다.)
# 
# 예를 들어, 아래 코드는 0부터 99,999 까지의 정수 중에서 3 또는 5의 배수를 모두 더한다.

# In[74]:


sum = 0

for i in range(100000):
    # %는 나머지 연산자
    if i % 3 == 0 or i % 5 == 0:
        sum += i
        
print(sum)


# **__`range()` 함수 주요 활용법 3__**

# `range()` 함수와 `list()`는 서로 함께 잘 활용된다.
# 먼저, `range()` 함수를 이용하여 `range` 객체를 생성한 다음에 바로 리스트로 변환하면
# 리스트를 간단하게 구현할 수 있다.

# In[75]:


gen = range(10)
gen


# In[76]:


list(gen)


# ## 순차 자료형에 유용한 함수

# 문자열, 튜플, 리스트처럼 항목들의 순서가 중요한 순차 자료형과 함께 
# 유용하게 사용되는 네 개의 함수를 소개한다.

# **`enumerate()` 함수**

# 튜플과 리스트의 인덱스를 튜플과 리스트 자체에서 눈으로 확인할 수 없다.
# 하지만 항목과 해당 항목의 인덱스 정보를 함께 활용해야 할 때가 있는데
# 이때 `enumerate()` 함수가 매우 유용하다.

# In[77]:


some_list = ['foo', 'bar', 'baz', 'pyt', 'thon']


# `enumerate()` 함수는 리스트를 받아서
# 리스트의 항목과 인덱스를 쌍으로 갖는 모음 자료형의 객체를 준비시킨다.
# 이렇게 준비된 객체를 직접 확인할 수는 없다.

# In[78]:


enumerate(some_list)


# 하지만 `for` 반복문을 이용하여 그 내용을 확인하고 활용할 수 있다.
# 예를 들어, 아래 코드는 짝수 인덱스의 값들만 출력하도록 한다.
# 
# **주의사항:** `i`와 `v` 두 변수를 활용하는 방식은 튜플 헤치기 방식이다.

# In[79]:


for i, v in enumerate(some_list):
    if i % 2 == 0:
        print(v)


# 아래 코드는 리스트의 항목을 키(key)로, 인덱스는 값(value)으로 하는 항목들로 이루어진 사전 자료형 객체를 
# 생성한다.

# In[80]:


mapping = {}

for i, v in enumerate(some_list):
    mapping[v] = i

mapping


# **`sorted()` 함수**

# `sorted()` 함수는 문자열, 튜플, 리스트의 항목을 크기 순으로 정렬시킨 리스트를 반환한다.

# In[81]:


sorted('horse race')


# In[82]:


sorted((7, 1, 2, 6, 0, 3, 2))


# In[83]:


sorted([7, 1, 2, 6, 0, 3, 2])


# **`zip()` 함수**

# 문자열, 튜플, 리스트 여러 개의 항목을 순서대로 짝지어서 튜플의 리스트 형식의 객체를 생성한다.
# 단, `zip()` 함수의 반환값은 `enumerate()`, `range()` 함수처럼 구체적으로 명시해주지는 않는다.

# In[84]:


zip("abc", "efg")


# 하지만 리스트로 변환하면 쉽게 내용을 확인할 수 있다.

# In[85]:


list(zip("abc", "efg"))


# 자료형이 달라도 되며, 각 자료형의 길이가 다르면 짧은 길이에 맞춰서 짝을 짓는다.

# In[86]:


list(zip("abc", [1, 2]))


# In[87]:


seq1 = ['foo', 'bar', 'baz']
seq2 = ['one', 'two', 'three']
zipped = zip(seq1, seq2)
list(zipped)


# 세 개 이상의 짝짓기도 가능하다.

# In[88]:


seq3 = [False, True]

list(zip(seq1, seq2, seq3))


# `enumerate()` 처럼 `for` 반복문에 잘 활용된다.
# 아래 코드는 두 개의 리스트의 항목을 짝을 지은 후 인덱스와 함께 출력해준다.

# In[89]:


for i, (a, b) in enumerate(zip(seq1, seq2)):
    print('{0}: {1}, {2}'.format(i, a, b))


# 동일한 인덱스에 위치한 항목들끼리 따로따로 모을 수 있다.

# In[90]:


pitchers = [('Nolan', 'Ryan'), ('Roger', 'Clemens'), ('Schilling', 'Curt')]

first_names, last_names = zip(*pitchers)


# In[91]:


first_names


# In[92]:


last_names


# 위 코드에서 사용된 별표(asterisk) 기호는 리스트를 해체하는 기능을 수행한다.
# 
# 즉, `pitchers`가 아래 리스트를 가리킬 때
# 
# ```python
# [('Nolan', 'Ryan'), ('Roger', 'Clemens'), ('Schilling', 'Curt')]
# ```
# 
# `*pitchers`는 다음 세 개의 튜플을 가리킨다.
# 
# ```python
# ('Nolan', 'Ryan'), ('Roger', 'Clemens'), ('Schilling', 'Curt')
# ```
# 
# 따라서 `zip(*pitchers)`는 다음과 같은 함수의 호출이 된다.

# In[93]:


first_names, last_names = zip(('Nolan', 'Ryan'), ('Roger', 'Clemens'), ('Schilling', 'Curt'))


# In[94]:


first_names


# In[95]:


last_names


# **`reversed()` 함수**

# 순차 자료형의 항목을 역순으로 갖는 순차 자료형을 생성한다.

# In[96]:


list(reversed(range(10)))


# In[97]:


list(reversed([3, 2, 5, 7]))


# In[98]:


list(reversed("abc"))


# In[99]:


list(reversed((1, 2, 3)))


# ## 사전

# 현대 프로그래밍 언어 분야에서 가장 중요하게 사용되는 자료형이 사전(`dict`)이다. 
# 특히, 데이터 분석 분야에서 더욱 그러하다. 
# 언어에 따라 해시맵(hash map), 연관배열(associative array) 등으로 불리기도 하며,
# 조금씩 다른 성질을 갖기도 하지만 기본적으로 파이썬의 사전 자료형과 동일하게 작동한다.
# 사전 자료형을 사람에 따라 딕셔너리(dictionary)라고 부르기도 하지만 여기서는 사전이라 부른다.

# 사전 자료형은 모음 자료형이며 따라서 여러 개의 항목을 갖는다.
# 각 항목은 __키(key)__와 __값(value)__의 쌍으로 이루어지며 아래 형식으로 키-값의 관계를 지정한다.
# 
# ```python
# 키(key) : 값(value)
# ```

# 사전 객체는 중괄호를 사용한다. 
# 집합에 사용되는 기호와 동일하지만 항목이 `키:값` 형식이라면 사전 객체로 인식된다.

# **빈 사전**

# 항목이 전혀 없는 빈 사전은 아래와 같이 표기한다.

# In[100]:


empty_dict = {}


# In[101]:


type(empty_dict)


# 반면에 공집합은 아래와 같이 선언한다.

# In[102]:


empty_set = set()


# In[103]:


type(empty_set)


# **사전의 항목 추가**

# 사전은 변경 가능하다.  
# 
# 예를 들어, `d1`은 아래와 같이 두 개의 항목을 갖는 사전을 가리킨다.

# In[104]:


d1 = {'a' : 'some value', 'b' : [1, 2, 3, 4]}

d1


# `7 : 'an integer'` 를 새로운 항목을 추가하려면 아래와 같이 진행한다.

# In[105]:


d1[7] = 'an integer'


# In[106]:


d1


# 다음은 `'language' : 'python'` 을 추가해보자.

# In[107]:


d1['language'] = 'python'

d1


# **사전의 항목 확인**

# 특정 키가 사전에 사용되었는지 여부를 확인할 때 `in` 연산자를 활용한다.

# In[108]:


'b' in d1


# 특정 키와 연관된 값을 확인하려면 인덱싱 방식처럼 사용한다.
# 단, 키를 인덱스 대신 지정하면 된다.

# In[109]:


d1['b']


# 없는 키의 값을 확인하려고 시도하면 `KeyError` 오류가 발생한다.

# ```python
# In [6]: d1['c']
# ---------------------------------------------------------------------------
# KeyError                                  Traceback (most recent call last)
# Input In [6], in <cell line: 1>()
# ----> 1 d1['c']
# 
# KeyError: 'c'
# ```

# 따라서 사전의 키-값을 확인할 때 발생할 수 있는 오류를 방지하기 위해 보통 
# 아래처럼 `if ... else ...` 조건문을 사용한다.

# In[110]:


if 'c' in d1: 
    print(d1['c'])
else:
    print("키가 없어요.")


# 그런데 `get()` 메서드는 대괄호 기호와 동일한 일을 하면서 오류를 발생시키지 않는다.

# In[111]:


d1.get('c')


# `get()` 메서드는 키가 존재하자 않으면 오류를 발생시키는 대신에 `None`을 반환한다.
# 또한, 키가 존재하지 않을 때 지정된 값을 반환하도록 할 수도 있다.
# 지정할 값을 둘째 인자로 정해놓으면 된다.

# In[112]:


d1.get('c', "키가 없어요")


# 결론적으로, 인덱싱 방식으로 키와 관련된 값을 확인하는 것 보다는 `get()` 메서드를 사용하면 
# 오류 발생 가능성을 줄일 수 있다.

# **`defaultdict` 클래스**

# 문자열 맨 처음에 위치한 알파벳을 기준으로 하여 문자열을 정리하고자 한다. 
# 주어진 단어는 다음과 같다.

# In[113]:


words = ['apple', 'bat', 'bar', 'atom', 'book']


# * 단어를 시작하는 알파벳 기준으로 구분하기 위해 문자열 인덱싱을 사용하여 첫 알파벳을 알아낸다. 
# * 알아낸 첫 알파벳을 키(key)로, 키의 값은 해당 알파벳으로 시작하는 단어들의 리스트이다.
#     예를 들어, 알파벳 `a` 로 시작하는 단어를 값으로 갖는 항목은 아래 모양이다.
#     
#     ```python
#     'a' : ['apple', 'atom']
#     ```

# 위 설명을 코드로 구현하면 다음과 같다.

# In[114]:


# 비어있는 사전 선언
by_letter = {}

# 모든 단어를 대상으로 첫 알파벳 확인 후 사전에 추가
for word in words:
    letter = word[0]

    if letter not in by_letter:      # letter로 시작하는 단어가 처음인 경우: 새로운 사전 항목 생성
        by_letter[letter] = [word]
    else:                            # letter로 시작하는 단어가 이미 이전에 리스트에 추가된 경우: 기존 리스트에 추가
        by_letter[letter].append(word)


# In[115]:


by_letter


# 잘 작동한다. 하지만 알파벳이 이미 사전에 키로 포함되어 있는가를 먼저 확인해야 하는 불편함이 존재한다.
# 만약에 `if ... else ...`를 사용하지 않으면 오류가 발생한다.

# ```python
# In [8]: by_letter = {}
# 
# In [9]: for word in words:
#    ...:     letter = word[0]
#    ...:     by_letter[letter].append(word)
#    ...:
# ---------------------------------------------------------------------------
# KeyError                                  Traceback (most recent call last)
# Input In [9], in <cell line: 1>()
#       1 for word in words:
#       2     letter = word[0]
# ----> 3     by_letter[letter].append(word)
# 
# KeyError: 'a'
# ```

# 이유는 `'apple'` 단어에서 `'a' : ['apple']`을 추가 해야 하는데
# `'a'`가 아직 키로 지정되지 않았기에 오류가 나는 것이다. 
# 
# 반면에 아래와 같이 하면 오류가 발생하진 않지만 제대로 작동하지 않는다.

# In[116]:


by_letter = {}

for word in words:
    letter = word[0]
    by_letter[letter] = [word]


# In[117]:


by_letter


# 이유는 새로운 단어로 매번 키의 값이 업데이트되기 때문이다.
# 이와 같이 키가 기존에 사용되었는지 여부를 매번 확인하는 불편함을 한 번에 해결하려면
# `collections` 모듈의 `defaultdict` 클래스를 활용한다.
# 즉, 굳이 키의 사용여부를 확인할 필요가 없다.
# 이유는 만약에 키로 사용된 적이 없다면 키의 값으로 비어있는 리스트를 만들어서 항목을 추가해주기 때문이다.

# In[118]:


from collections import defaultdict

by_letter = defaultdict(list)

for word in words:
    by_letter[word[0]].append(word)
    
by_letter


# 사전 자료형의 `setdefault()` 메서드가 유사한 기능을 지원한다.
# 하지만 `defaultdict` 클래스를 보다 많이 사용한다.

# In[119]:


by_letter = {}

for word in words:
    letter = word[0]
    by_letter.setdefault(letter, []).append(word)
    
by_letter


# **사전의 항목 삭제**

# `del` 예약어와 `pop()` 메서드를 이용하여 특정 키가 사용된 항목을 삭제할 수 있다.

# In[120]:


d1[5] = 'some value'
d1['dummy'] = 'another value'


# In[121]:


d1


# `del` 예약어는 함수가 아니라 파이썬 자체에서 지원하는 특별한 기능을 가진 명령문이다. 
# 아래 명령문은 5를 키워드로 갖는 `5: 'some value'`를 사전에서 삭제한다.

# In[122]:


del d1[5]

d1


# 반면에  `pop()` 메서드는 지정 항목을 삭제하면서 동시에 지정된 키와 연관된 값을 반환한다.

# In[123]:


d1.pop('dummy')


# In[124]:


d1


# **`keys()` 메서드**

# 키만 모아 놓은 리스트를 구할 수 있다.

# In[125]:


list(d1.keys())


# **`values()` 메서드**

# 값만 모아 놓은 리스트를 구할 수 있다.

# In[126]:


list(d1.values())


# **사전 합치기**

# 하나의 사전에 포함된 항목 전체를 다른 사전에 추가할 수 있다.
# 아래 코드는 `d1` 사전에 두 개의 항목을 추가한다.
# 
# __주의사항:__ 동일한 키가 추가될 경우 기존에 사용된 값이 새로운 값으로 업데이트 된다.

# In[127]:


d1.update({'b' : 'foo', 'c' : 12})

d1


# **사전의 항목 업데이트**

# 기존에 포함된 키-값 에서 값을 변경하려면 다음과 같이 한다.
# 리스트에서 항목을 수정하는 방식과 유사하며,
# 인덱스 대신에 키를 이용한다.
# 예를 들어, 아래 코드는 `'language'`의 값을 `'python'` 에서 `'python3'`로 업데이트한다.

# In[128]:


d1['language'] = 'python3'

d1


# **`dict()` 함수**

# 모든 항목이 길이가 2인 튜플 또는 리스트인 모음 자료형을 인자로 사용하여
# 새로운 사전을 생성한다.

# In[129]:


dict([(1, 'a'), (2, 'b')])


# In[130]:


dict(([1, 'a'], [2, 'b']))


# `zip()` 함수를 이용하여 두 개의 리스트 또는 튜플을 엮어 사전을 쉽게 생성할 수 있다.

# In[131]:


mapping = dict(zip(range(5), reversed(range(5))))
mapping


# **사전의 키로 사용될 수 있는 자료형**

# 변경 불가능한 객체만 사전의 키로 사용될 수 있다.
# 예를 들어, 문자열, 정수, 실수, 튜플 등이다. 
# 단, 튜플의 항목에 리스트 등 변경 가능한 값이 사용되지 않아야 한다. 
# 
# 이렇게 사전의 키로 사용될 수 있는 값은 __해시 가능__(hashable)하다고 하며
# `hash()` 함수를 이용하여 해시 가능 여부를 판단할 수 있다.
# `hash()` 함수의 반환값은 두 종류이다. 
# 
# * 해시 가능일 때: 특정 정수
# * 해시 불가능일 때: 오류 발생

# 문자열과 정수로만 이루어진 튜플은 해시 가능이다.

# In[132]:


hash('string')


# In[133]:


hash((1, 2, (2, 3)))


# 따라서 튜플도 사전의 키로 사용할 수 있다.

# In[134]:


{(1, 2, (2, 3)) : "튜플 사용 가능"}


# 반면에 리스트를 포함한 튜플은 해시 불가능이다.

# ```python
# In [10]: hash((1, 2, [2, 3]))
# ---------------------------------------------------------------------------
# TypeError                                 Traceback (most recent call last)
# Input In [10], in <cell line: 1>()
# ----> 1 hash((1, 2, [2, 3]))
# 
# TypeError: unhashable type: 'list'
# ```

# 따라서 `(1, 2, [2, 3])`을 키로 사용하면 오류가 발생한다.

# ```python
# In [11]: {(1, 2, [2, 3]) : "오류 발생"}
# ---------------------------------------------------------------------------
# TypeError                                 Traceback (most recent call last)
# Input In [11], in <cell line: 1>()
# ----> 1 {(1, 2, [2, 3]) : "오류 발생"}
# 
# TypeError: unhashable type: 'list'
# ```

# ## 집합

# 집합 자료형은 수학에서 배운 집합과 동일한 개념이다.
# 중괄호 기호를 사용하지만 사전 자료형과 혼동되지는 않을 것이다.
# 집합의 항목들 사이에는 순서가 없으며, 중복도 허용하지 않는다.
# 참고로 사전과 집합은 순차 자료형이 아니다.

# In[135]:


{2, 2, 2, 1, 3, 3}


# 순서와 원소의 중첩 여부와 상관 없이 동일한 원소를 포함하면 동일한 집합으로 간주한다.

# In[136]:


{1, 2, 3, 1} == {3, 2, 1}


# **`set()` 함수**

# `set()` 함수를 이용하여 리스트, 튜플 등을 집합으로 변환시킬 수 있다.

# In[137]:


set([2, 2, 2, 1, 3, 3])


# In[138]:


set((2, 2, 2, 1, 3, 3))


# 이 기법은 리스트와 튜플에서 중복된 항목을 제거하고자 할 때 유용하다.

# In[139]:


list(set([2, 2, 2, 1, 3, 3]))


# In[140]:


tuple(set((2, 2, 2, 1, 3, 3)))


# **집합의 항목 추가/삭제**

# 집합은 변경이 가능하다. 
# 항목 추가는 `add()` 메서드를 활용한다.

# In[141]:


a_set = {1, 2, 3}
a_set.add(4)

a_set


# 항목 삭제는 `remove()` 메서드를 이용한다.
# `remove()` 메서드는 원소를 삭제하지만, 삭제된 값을 반환하지는 않는다.
# 실제 반환값은 `None`이다.

# In[142]:


a_set.remove(4)


# In[143]:


a_set


# 없는 항목을 삭제하려 하면 오류가 발생한다.

# ```python
# In [13]: a_set.remove(4)
# ---------------------------------------------------------------------------
# KeyError                                  Traceback (most recent call last)
# Input In [13], in <cell line: 1>()
# ----> 1 a_set.remove(4)
# 
# KeyError: 4
# ```

# **합집합 연산**

# `union()` 메서드는 두 집합의 합집합을 반환한다.
# 
# __참고:__ 엄밀히 따지면 `a.union(b)`는 집합 `a`에 집합 `b`의 원소를 추가하는 방식으로
# 새로운 집합을 생성한다.

# In[144]:


a = {1, 2, 3, 4, 5}
b = {3, 4, 5, 6, 7, 8}


# In[145]:


a.union(b)


# 이항 연산자 `|`가 합집합 연산을수행한다.

# In[146]:


a | b


# **교집합 연산**

# `intersection()` 메서드는 두 집합의 교집합을 반환한다.
# 
# __참고:__ 엄밀히 따지면 `a.intersection(b)`는 집합 `a`의 원소 중에서 집합 `b`에 속한
# 원소만을 모아 새로운 집합을 생성한다.

# In[147]:


a.intersection(b)


# 이항 연산자 `&`가 교집합 연산을수행한다.

# In[148]:


a & b


# 합집합, 교집합 연산은 기존의 집합은 변경하지 않으면서 새로운 집합을 생성한다.

# In[149]:


a


# In[150]:


b


# **부분집합 여부 판단**

# `issubset()` 메서드를 이용한다.

# In[151]:


a_set = {1, 2, 3, 4, 5}
{1, 2, 3}.issubset(a_set)


# `issuperset()` 메서드는 상위집합(superset) 여부를 판단한다.

# In[152]:


a_set.issuperset({1, 2, 3})


# **집합 원소의 자료형**

# 사전의 키의 경우처럼 집합의 원소는 모두 해시 가능이어야 한다.
# 즉, 리스트는 집합의 원소가 될 수 없다.

# ```python
# In [14]: my_data = [1, 2, 3, 4]
# 
# In [15]: my_set = {my_data}
# ---------------------------------------------------------------------------
# TypeError                                 Traceback (most recent call last)
# Input In [15], in <cell line: 1>()
# ----> 1 my_set = {my_data}
# 
# TypeError: unhashable type: 'list'
# ```

# (sec:list-comprehension)=
# ## 조건제시법

# 리스트, 집합, 사전을 수학 시간에 배운 조건제시법(comprehension)을 이용하여 정의할 수 있다.
# 이 기법은 특히 리스트와 함께 매우 유용하게 활용된다.

# ### 리스트 조건제시법

# 예를 들어, 0부터 9 까지의 자연수 중에서 짝수로 이루어진 집합을 수학에서 
# 조건제시법으로 아래와 같이 정의한다.
# 
# ```
# { x | 0 < x < 10, 단 x는 짝수 } = { 0, 2, 4, 6, 8 }
# ```
# 
# 여기서 집합 기호를 대괄호로 바꾸면 거의 바로 리스트 조건제시법이 된다.

# In[153]:


a_list = [x for x in range(0, 10) if x%2 == 0]

a_list


# '위 조건제시법은 아래 `for` 반복문을 활용한 아래 코드와 동일하다.

# In[154]:


a_list = []

for x in range(0, 10):
    if x%2 == 0:
        a_list.append(x)
        
a_list


# __예제__
# 
# 문자열로 이루어진 리스트를 이용하여 모두 대문자로 전환된 문자열들의 리스트를 생성할 수 있다.
# 단, 문자열의 길이가 2보다 커야 한다.
# 
# * '단', 즉, 조건에 해당하는 부분은 `if` 조건문으로 처리한다.

# In[155]:


strings = ['a', 'as', 'bat', 'car', 'dove', 'python']

[x.upper() for x in strings if len(x) > 2]


# ### 집합 조건제시법

# 집합에 대한 조건제시법 사용도 유사하다.
# 아래 코드는 앞서 사용된 문자열들의 길이를 원소로 갖는 집합을 생성한다.

# In[156]:


unique_lengths = {len(x) for x in strings}

unique_lengths


# ### 사전 조건제시법

# 조건제시법을 이용하여 사전을 생성하는 과정도 유사하다.
# 아래 코드는 앞서 사용된 문자열을 키로, 문자열의 길이를 값으로 하는 사전을 생성한다.

# In[157]:


len_mapping = {val : len(val) for val in strings}

len_mapping


# 아래 코드는 앞서 사용된 문자열을 키로, 문자열의 인덱스를 값으로 하는 사전을 생성한다.

# In[158]:


loc_mapping = {val : index for index, val in enumerate(strings)}

loc_mapping


# ### 중첩 조건제시법

# **예제**

# 아래 리스트는 중첩 리스트이다.

# In[159]:


all_data = [['John', 'Emily', 'Michael', 'Mary', 'Steven'],
            ['Maria', 'Juan', 'Javier', 'Natalia', 'Pilar']]


# `all_data`에 포함된 이름 중에서 알파벳 `n`이 사용된 이름으로만 구성된 리스트를 작성하고자 한다.
# 먼저, 각 리스트에서 조건을 만족하는 이름으로 구성된 리스트를 조건제시법으로 구현하면 다음과 같다.
# 
# 먼저, 첫째 리스트를 대상으로 한다.

# In[160]:


[name for name in all_data[0] if name.count('n') >= 1]


# 둘째 리스트가 대상이면 다음과 같다.

# In[161]:


[name for name in all_data[1] if name.count('n') >= 1]


# 위 과정을 한 번에 진행하려면 아래와 같이 `for` 반복문을 이용하면 된다.
# 단, 이번에는 이름을 담을 리스트를 미리 준비한다. 
# 
# * `all_data[i]` 대신에 `item`을 사용하여 `all_data` 의 항목을 순환하도록 한다.

# In[162]:


result = []

for item in all_data:
    result.extend([name for name in item if name.count('n') >= 1])
    
result


# 이와 같이 `for` 반복문 안에 리스트 조건제시법이 사용된 경우 
# 이중 조건제시법을 사용할 수 있다.

# In[163]:


result = [name for item in all_data for name in item if name.count('n') >= 1]

result


# **예제**

# 중첩 리스트, 중첩 튜플, 또는 아래와 같이 튜플과 리스트가 중첩으로 사용된 경우
# 모든 중첩을 제거하고 사용된 항목으로만 이루어진 리스트 또는 튜플을 생성할 때
# 중첩 조건제시법이 매우 유용하다.
# 이렇게 중첩 사용된 모음 자료형을 1차원 리스트로 단순화 시키는 작업을 영어로 **flatten**이라 한다.

# In[164]:


some_tuples = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]


# In[165]:


flattened = [x for tup in some_tuples for x in tup]

flattened


# 위 조건제시법이 작동하는 과정을 아래 `for` 반복문이 설명한다.

# In[166]:


flattened = []

for tup in some_tuples:
    for x in tup:
        flattened.append(x)


# **예제**

# 아래 코드는 항목으로 사용된 튜플을 모두 리스트로 변환하여 중첩 리스트를 생성한다.

# In[167]:


[[x for x in tup] for tup in some_tuples]


# ## 연습문제

# 1. [(실습) 모음 자료형 1부: 문자열, 리스트, 튜플](https://colab.research.google.com/github/codingalzi/pybook/blob/master/practices/practice-collections1.ipynb)
# 1. [(실습) 모음 자료형 2부: 집합, 사전, range, 조건제시법](https://colab.research.google.com/github/codingalzi/pybook/blob/master/practices/practice-collections2.ipynb)
