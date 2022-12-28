#!/usr/bin/env python
# coding: utf-8

# # 자료형 명시(Type Annotation)

# ## 파이썬과 동적 타이핑

# 파이썬은 **동적 타이핑**(dynamic typing)을 지원하는 언어이다.
# 즉, 함수나 변수를 선언할 때 변수들의 자료형을 명시적으로 제한하지 않는다.
# 동적 타이핑 언어의 경우 프로그램 실행 과정에서 문제가 발생하지 않도록 프로그램을 작성해야 한다.
# 
# 예를 들어 아래 `add` 함수를 보자.

# In[1]:


def add(a, b):
    return a + b


# `add` 함수의 인자로 정수, 실수, 리스트, 문자열이 사용될 수 있다.

# In[2]:


assert add(10, 5) == 15,                  "정수들에 대해 + 사용 가능"
assert add([1, 2], [3]) == [1, 2, 3],     "리스트들에 대해 + 사용 가능"
assert add("저 ", "잠깐만요!") == "저 잠깐만요!", "문자열들에 대해 + 사용 가능"


# 하지만 두 인자가 동일한 자료형을 가져야 한다.
# 예를 들어, 숫자와 문자열의 덧셈은 작동하지 않는다.
# 이유는 숫자와 문자열의 덧셈이 정의되어 있지 않기 때문이다.

# In[3]:


try:
    add(10, "five")
except TypeError:
    print("정수와 문자열은 서로 더할 수 없어요!")


# ## 파이썬과 정적 타이핑

# C, Java 등 많은 프로그래밍 언어는 동적 타이핑 대신에 **정적 타이핑**(static typing)을 지원한다.
# 즉, 함수나 변수를 선언할 때 사용되는 변수들의 자료형과 인자 및 반환값의 자료형을 
# 애초부터 명시해야 하며 지정된 자료형이 사용되지 않을 경우 오류를 발생시킨다.
# 
# 파이썬은 3.6 버전부터 정적 타이핑 형식을 지원한다. 
# 다만 C, Java의 자료형과 관련된 엄격함은 전혀 존재하지 않으며, 
# 그냥 정적 타이핑의 형식만 빌려왔다.
# 
# 즉, **자료형 명시**(type annotations)를 지원할 뿐이며, 
# 실제로는 동적 타이핑 방식을 사용한다.
# 예를 들어, `add` 함수를 아래와 같이 선언할 수 있다.

# In[4]:


def add(a: int, b: int) -> int:
    return a + b


# 하지만 여전히 문자열이나 리스트를 인자로 사용할 수 있다.

# In[5]:


print(add(10, 5))
print(add([1, 2], [3]))
print(add("저 ", "잠깐만요!"))


# ## 자료형 명시의 장점

# 비록 형식적더라도 자료형 명시하기가 주는 장점이 크게 네 가지이다.

# 첫째, 문서화 및 프로그래밍 교육에 유용하다.
# 
# 예를 위해, 먼저 벡터 자료형을 실수들의 리스트들의 집합으로 정의하자.
# 
# * 기본 자료형의 정의는 타이핑(`typing`) 모듈에 포함되어 있다.
# * 벡터(Vector) 자료형은 실수들의 리스트, 즉, `List[float]`로 정의된다.

# In[6]:


from typing import List
Vector = List[float]


# 이제 아래 두 개의 정의를 비교하면 둘째 정의가 보다 많은 정보를 우리에게 제공함을 알 수 있다.
# 
# 정의 1: 전통적 방식

# In[7]:


def dot_product(x, y): ...


# 정의 2: 자료형 명시

# In[8]:


def dot_product(x: Vector, y: Vector) -> float: ...


# 둘째, `mypy` 와 같은 제3자가 개발한 툴을 이용하여 파이썬 코드를 실행하기 전에
# 작성된 코드에 사용된 함수와 변수들이 적절한 자료형을 사용했는지 여부를
# 검사해주는 툴을 활용할 수 있다.
# 
# 하지만 여기서는 사용하지 않을 것이며, 대신에 관심이 있다면 
# [mypy 공식 문서](https://mypy.readthedocs.io/en/stable/)를 참조하기를 추천한다.

# 셋째, 자료형을 명시적으로 보여줌으로써 보다 정제된 함수와 인터페이스를 디자인할 수 있다.
# 
# 예를 들어, 아래 `secretly_ugly_function`인 경우 `value`와 `operation` 매개변수에
# 사용할 인자들의 자료형을 함수의 본체를 들여다보기 전까지는 전혀 알 수 없다.

# In[9]:


def secretly_ugly_function(value, operation): ...


# 아래 `ugly_function` 또한 여전히 부자연스럽다.

# In[10]:


from typing import Union

def ugly_function(value: int, operation: Union[str, int, float, bool]) -> int:
    ...


# 왜냐하면 둘째 매개변수 `operation`에 할당될 수 있는 함수는 인자로 
# `str`, `int`, `float`, `bool` 중의 하나의 자료형을 받아들일 수 있기 때문이다.
# 그래도 이렇게 자료형을 명시하면 
# 사용자들이 프로그램을 보다 쉽게 이해할 수 있도록 도와준다.
# 
# **주의:** `Union`은 합집합을 나타내는 기호이다.

# 넷째, 소스코드 에디터의 자동완성 기능을 보다 적절하게 지원하게 만들어 주며,
# 따라서 보다 빠르게 오류 없는 프로그램을 작성할 수 있도록 도와준다. 
# 
# 예를 들어, VSCode 에디터에서 아래 코드를 작성하다 보면 
# 매개변수 `xs`가 정수들의 리스트를 입력받을 것으로 기대하며,
# 리스트의 메소드의 목록을 보여주며 코드 작성을 도와주려 시도한다.
# (구글 코랩은 아직 지원하지 않음.)

# <img src="https://raw.githubusercontent.com/codingalzi/pydata/master/notebooks/images/type_annotation.png" width="50%">

# ## 자료형 명시 방법

# * 내장 자료형(built-in types): `int`, `bool`, `float`, `str` 등은 그대로 사용
# * `a:int`, `b:bool`, `c:float`, `s:str` 등등

# * 리스트: 단순히 `list` 라고 하는 것은 별 도움 않됨. 

# In[11]:


def total(xs: list) -> float:
    return sum(xs)


# * 대신에 어떤 자료형의 리스트인지 명시하는 게 보다 유용

# In[12]:


from typing import List  # 대문자 L 사용에 주의할 것

def total(xs: List[float]) -> float:
    return sum(xs)


# ## 변수 자료형 명시

# 변수의 자료형도 명시할 수 있다.

# In[13]:


x: int = 5


# 경우에 따라 변수의 자료형이 명확하지 않아서 자료형을 명시하면 많은 도움을 받을 수 있다.
# 예를 들어, 아래 두 변수는 어떤 종류의 리스트인지 명확하지 않다.

# In[14]:


values = []


# 대신에 아래의 경우는 아주 명확하다.
# 즉, 빈 리스트이면서 정수들의 리스트 중에 하나임을 명확히 보여준다.

# In[15]:


values : List[int]= []


# ## `Optional` 자료형

# 아래의 경우는 애매함이 더욱 심하다.

# In[16]:


best_so_far = None


# `None`은 '아무 값도 아니다'를 가리키는 '값'이다. 
# 하지만 변수는 어떤 값을 가리키기 위해 존재하며, 새로운 값을 언제라도 가리킬 수 있다.
# 따라서 `None`이 언제라도 다른 값으로 대체될 수 있으며,
# 대체될 값의 자료형을 암시해줄 필요가 있다.
# 이를 위해 옵셔널(`Optional`) 자료형을 `List` 자료형과 유사한 방법으로 활용할 수 있다.

# In[17]:


from typing import Optional

best_so_far: Optional[float] = None


# 위와 같이 하면 `best_so_far` 변수에는 실수(`float`) 자료형이 할당될 것으로 
# 기대함을 바로 알 수 있다.
# 즉, `float` 자료형 또는 `None` 값을 위해 `best_so_far` 변수를 사용할 것이라고
# 명시하는 것이다.

# ## `typing` 모듈

# 타이핑(`typing`)은 `List`, `Optional` 이외에도 다른 많은 자료형을 포함하고 있다.
# 그중에 일부만 다룰 예정이다.

# In[18]:


from typing import Dict, Iterable, Tuple


# `counts` 변수는 문자열을 키(key)로, 정수를 키값으로 사용하는 사전 자료형을 담고 있다.

# In[19]:


counts: Dict[str, int] = {'data': 1, 'science': 2}


# `evens` 변수는 0부터 9사이의 짝수를 소극적(lazy) 리스트로 담고 있다. 

# In[20]:


evens: Iterable[int] = (x for x in range(10) if x % 2 == 0)


# `triple` 변수는 정수, 실수, 정수 세 개의 값을 갖는 튜플이다.

# In[21]:


triple: Tuple[int, float, int] = (10, 2.3, 5)


# ## `Callable` 함수의 자료형

# 파이썬에서 함수는 제1종 객체이다. 
# 즉, 함수의 인자로 함수를 입력할 수 있다.
# 따라서 함수를 가리키는 매개변수의 자료형도 명시할 수 있어야 한다. 
# 이를 위해 `Callable`(호출가능한) 자료형을 활용한다.

# In[22]:


from typing import Callable

def twice(repeater: Callable[[str, int], str], s: str) -> str:
    return repeater(s, 2)


# * `twice` 함수는 첫째 인자로 `Callable[[str, int], str]` 자료형의 함수를 인자로 받는다.
# * `Callable[[str, int], str]` 는 아래 특징을 갖는 함수들의 자료형을 가리킨다.
#     * 인자 함수 입력값 두 개
#         * 첫째 인자: 문자열(`str`)
#         * 둘째 인자: 정수(`int`)
#     * 인자 함수 반환값: 문자열(`str`)

# 예를 들어, 아래 함수 `comma_repeater`를 `twice` 함수의 인자로 입력해보자.

# In[23]:


def comma_repeater(s: str, n: int) -> str:
    n_copies = [s for _ in range(n)]
    return ', '.join(n_copies)


# In[24]:


comma_repeater("type hints", 2)


# `twice(comma_repeater, "type hints")`는 
# `comma_repeater("type hints", 2)`의 반환값을 반환한다.

# In[25]:


twice(comma_repeater, "type hints") == comma_repeater("type hints", 2)


# ## 자료형 객체

# 자료형을 명시하기 위해 사용된 자료형들 자체도 파이썬 객체이다.
# 즉, 변수에 할당할 수 있는 값으로 사용될 수 있다.
# 
# 예를 들어, 복잡한 자료형을 단순한 이름으로 지정하여 다시 자료형 명시를 위해 사용할 수 있다.

# In[26]:


Number = int
Numbers = List[Number]

def total(xs: Numbers) -> Number:
    return sum(xs)

