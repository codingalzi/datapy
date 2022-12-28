#!/usr/bin/env python
# coding: utf-8

# # 클래스와 자료형

# 파이썬은 소위 **객체 지향 프로그래밍**을 지원하는 언어이다.
# 이 강좌에서는 객체 지행 프로그래밍의 정체를 논하지 않는다.
# 다만, 객체 지향 프로그래밍의 핵심 요소인 클래스(class)와 인스턴스(instance)를 
# 어떻게 정의하고 활용하는지 예제 두 개를 이용하여 보여준다.
# 
# 앞으로 클래스와 인스턴스를 수 없이 보고 사용할 것이다.
# 사실, 지금까지도 많은 클래스와 인스턴스를 살펴 보았다.
# 
# * `str`: 문자열 클래스
#     * 인스턴스: `"abc"`, `"홍길동"` 등
# 
# 
# * `list`: 리스트 클래스
#     * 인스턴스: `[1, 2, 3]`, `['ab', 1, [0, 1]]` 등
# 
# 
# * `set`: 집합 클래스
#     * 인스턴스: `set(), {1, 2, 3}`, `{'ab', 1, [0, 1]}` 등
# 
# 
# * `dict`: 사전 클래스
#     * 인스턴스: `{"이름":"홍길동", "출신":"한양"}` 등

# __클래스의 활용__
# 
# 클래스는 크게 세 가지 방식으로 사용된다.
# 
# 1. 자료형 및 관련 메서드 정의: 여기에서 주로 사용되는 방식이며, 앞서 언급한 클래스들이 여기에 해당한다. 
# 1. 서로 관련된 기능을 갖는 변수와 함수들의 모둠, 일종의 도구 상자
# 1. 동일한 기능을 갖는 객체를 쉽고 다양한 방식으로 생성할 수 있도록 도와주는 기계틀 역할을 수행하는 도구:
#     게임 프로그래밍 등에서 게임 캐릭터, 배경, 도구 등을 쉽게 생성할 때 기본적으로 사용되는 방식이며,
#     여기서는 다루지 않는다.

# ## A. 자료형과 메서드 정의 예제: 집합 클래스 구현하기

# 집합 자료형인 `set`이 없다고 가정하고 직접 집합 자료형을 `MySet` 클래스를 이용하여 정의해보자.

# In[1]:


class MySet: 
    """집합 자료형을 간단한 기능과 함께 직접 구현한다.
    """

    # 생성자
    def __init__(self, values=None):
        """초기 설정 메서드, 소위 생성자. 
        자바의 경우와는 달리 모든 클래스의 초기 설정 메서드는 동일한 이름 사용
        아래 형식으로 활용:
        s1 = MySet()               # 공집합 생성
        s2 = MySet([1,2,2,3])      # 원소를 지정하면서 집합 생성
        """

        self.dict = {}            # 사전 객체를 이용하여 원소 저장
            
        if values is not None:    # values에 이터러블 자료형 기대
            for value in values:
                self.add(value)   # 아래에 정의된 add() 메서드 활용

    def __repr__(self):
        """MySet 객체를 출력할 때 사용되는 메서드
        """
        return "Set(" + str(set(self.dict.keys())) + ")"
    
    # 원소 추가
    def add(self, value): 
        self.dict[value] = True        # 항목은 키(key), 해당 값은 True
        
    # 원소 포함여부
    def isin(self, value): 
        return value in self.dict
    
    # 원소 제거
    def remove(self, value): 
        del self.dict[value]


# ### 매직 메서드

# 이닛(`__init__()`), 레퍼(`__repr__()`) 메서드처럼 두 개의 밑줄(underscores)로 감싸인 메서드를
# **매직 메서드**(magic methods)라 부르며, 모든 파이썬 클래스에 동일한 이름으로 포함되어 있다.
# 기타 많은 매직 메서드가 존재하며 명시적으로 선언되지 않은 매직 메서드는 모든 클래스에서 
# 동일하게 사용되는 기본 함수가 자동으로 지정된다. (대부분은 아무 것도 하지 않는 함수로 지정됨.)

# ### 클래스와 인스턴스

# 1, 2, 3을 원소로 갖는 `MySet`을 다음처럼 리스트를 이용하여 생성한다.

# In[2]:


s = MySet([1,2,3]) 


# 사실 거의 임의의 이터러블 자료형을 이용하여 `MySet` 객체를 만들 수 있다.
# 아래 코드는 집합을 이용한다.

# In[3]:


s = MySet((1,2,3)) 


# 여기서 변수 `s`에 할당된 값을 `MySet` 클래스의 객체 또는 인스턴스라 부르며,
# 이 경우에는 특별히 `MySet` 자료형이라 부를 수 있다.
# `set` 클래스의 인스턴스를 집합 자료형이라 부르는 것과 동일하다.

# ### `__init__()` 함수의 역할

# `MySet` 클래스의 객체를 선언하면 
# 파이썬 내부에서 `__init__()` 메서드가 호출된다.
# 
# * `self`에 대한 인자 지정하지 않아도 내부적으로 생성된 객체를 가리키는 역할이 지정됨.
# * `Values=None` 키워드 인자 사용

# ### 기타 메서드 활용

# 이제 `MySet` 자료형인 `s`에 원소를 추가/삭제하는 방법과 결과를 확인해보자.

# In[4]:


s


# In[5]:


print(s)


# In[6]:


s.add(4)


# In[7]:


s.isin(4)


# In[8]:


s.remove(3)


# In[9]:


s


# In[10]:


s.isin(3)


# ## B. 도구 상자 모둠 예제: 클릭수 세기

# 웹페이지의 방문자 수를 확인하는 앱을 구현하고자 할 때 아래 도구들이 필요하다.
# 
# - 클릭수를 저장할 변수
# - 클릭수를 1씩 키워주는 도구
# - 현재 클릭수를 읽어주는 도구
# - 클릭수를 초기화하는 도구
# 
# 언급한 변수 한 개와 네 개의 도구를 포함한 일종의 도구상자를
# 아래 `CountingClicker` 클래스로 구현한다.

# In[11]:


class CountingClicker:
    """ 웹페이지 방문자 수 확인 도구 모음 클래스
        - 클릭수를 저장할 변수
        - 클릭수를 1씩 키워주는 도구
        - 현재 클릭수를 읽어주는 도구
        - 클릭수를 초기화하는 도구
    """

    # 클래스 변수: 인스턴스 생성 횟수를 기억함.
    _total_count = 0

    # 생성자: 클릭수 초기값을 0으로 지정
    def __init__(self, _count = 0):
        self._count = _count                  # 인스턴스 변수
        CountingClicker._total_count += 1

    def __repr__(self):
        return f"CountingClicker(count={self._count})"
    
    def click(self, num_times = 1):
        """클릭할 때마다 카운드 올리기"""
        self._count += num_times

    # 현재 클릭수를 읽어주는 도구 역할 함수
    @property                               # 속성 변수 취급
    def read_count(self):
        return self._count
    
    # 클릭수를 초기화하는 도구 역할 함수
    def reset(self):
        self._count = 0
        
    # 클릭수를 초기화하는 도구 역할 함수
    @classmethod
    def reset_total(cls):
        cls._total_count = 0        


# 아래 코드를 실행하면 `CountingClass`의 인스턴스를 하나 생성하면 앞서 언급한 변수 한 개와 네 개의 도구를 
# 포함한 하나의 도구상자를 얻게 되며, 도구상자의 이름은 `clicker`이다.

# In[12]:


clicker = CountingClicker()


# `clicker` 도구상자에 포함된 도구, 즉 특정 메서드를 이용하려면 아래와 같이 실행한다.
# 
# ```python
# clicker.메서드이름(인자,....)
# ```

# 먼저 클릭수가 0으로 초기화되어 있음을 확인하자.
# 이유는 아무도 클릭하지 않았기 때문이다.
# 실제로 `clicker` 도구상자를 생성할 때 호출되는 `__init__` 메서드의 인자가 지정되지 않아서
# `count` 매개변수의 키워드 인자로 기본값인 0이 사용되어, `self.count` 변수에 할당되었다.

# In[13]:


assert clicker.read_count == 0, "클릭수는 0부터 시작해야 합니다."


# ### 인스턴스 변수
# 
# * `self.count`와 같은 변수를 **인스턴스 변수**라 부른다.
# * 이유는 인스턴스가 생성되어야만 존재의미를 갖는 변수이기 때문이다.
# * 인스턴스 변수 이외에 **클래스 변수**가 클래스를 선언할 때 사용될 수 있다.
#     클래스 변수는 인스턴스가 생성되지 않아도 존재의미를 갖는다.
#     이에 대해서는 나중에 기회 있을 때 자세히 설명한다.

# 이제 클릭을 두 번 했다가 가정하자. 
# 즉, `click` 메서드를 두 번 호출되어야 한다.

# In[14]:


clicker.click()
clicker.click()


# 그러면 클릭수가 2가 되어 있어야 한다.

# In[15]:


assert clicker.read_count == 2, "두 번 클릭했으니 클릭수는 2이어야 함."


# 이제 클릭수를 초기화하자.

# In[16]:


clicker.reset()


# 그러면 클릭수가 다시 0이 되어야 한다.

# In[17]:


assert clicker.read_count == 0, "클릭수를 초기화하면 0이 된다."


# 클릭수를 지정하면서 `CountingClicker`의 인스턴스를 생성할 수 있다.

# In[18]:


clicker50 = CountingClicker(50)


# 클릭수가 50으로 설정되었음을 확인할 수 있다.

# In[19]:


clicker50.read_count


# 지금까지 `CountingClicker`의 인스턴스는 두 번 생성되었음을 아래 결과가 보여준다.

# In[20]:


CountingClicker._total_count


# ### 클래스 상속

# 부모 클래스로부터 모든 기능을 물려받을 수 있는 자식 클래스를 상속을 이용하여 선언할 수 있다.
# 
# 예를 들어, 클릭수를 초기화할 수 없는 자식 클래스 `NoResetClicker`를 아래와 같이 선언한다.
# 클릭수 초기화 기능을 없애기 위해서는 `reset` 메서드를 재정의(overriding)해야 한다.

# In[21]:


class NoResetClicker(CountingClicker):
    # 부모 클래스인 CountingClicker의 모든 메서드를 물려 받는다.
    # 다만, read_count 함수와 reset 함수를 오버라이딩(재정의) 한다.
    # 재정의하지 않는 메서드는 부모 클래스에서 정의된 그대로 물려 받는다.
    
    # 부모 클래스의 read_count 함수의 반환값을 두 배해서 반환한다.
    # 부모 클래스를 가리키는 super()의 용법에 주의한다.
    
    @property
    def read_count(self):
        return 2 * (super().read_count)

    # reset 함수의 초기화 기능을 없앤다.
    def reset(self):
        pass
    


# 이제 `NoResetClicker`의 인스턴스를 생성한 후 클릭수 초기화가 이루어지지 않을 확인할 수 있다.

# In[22]:


clicker2 = NoResetClicker()
clicker2.click()
clicker2.click()
clicker2.read_count


# 리셋이 작동하지 않습니다.

# In[23]:


clicker2.reset()

clicker2.read_count


# 자식 클래스의 인스턴스가 만들어져도 부모 클래스의 인스턴스 카운트가 올라간다.

# In[24]:


CountingClicker._total_count


# 객체 생성횟수를 초기화하면 `_total_count`의 값이 0으로 된다.

# In[25]:


CountingClicker.reset_total()


# In[26]:


CountingClicker._total_count

