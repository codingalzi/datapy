#!/usr/bin/env python
# coding: utf-8

# (sec:python_basic_7)=
# # 파일

# 데이터 분석에서 가장 많이 다루는 파일은 크게 csv 이거나 엑셀 두 종류이며
# 판다스 모듈의 적절한 함수를 이용하여 불러올 수 있다.
# 하지만 여기서는 임의의 파일을 다운로드, 불러오고, 열어보는 기본적인 방식을 살펴본다.

# ## 파일의 종류와 경로

# **텍스트 파일과 바이너리 파일**
# 
# 파일은 **텍스트**<font size='2'>text</font> 파일과 
# **바이너리**<font size='2'>binary</font> 파일 두 가지로 나뉜다.
# 
# 바이너리 파일은 한컴 오피스의 한글 파일, MS 워드 파일과 파워포인트 파일,
# jpg, png, gif 등의 사진 파일 등 특정 소프트웨어를 사용해야만 
# 내용을 확인할 수 있는 파일이다.
# 그리고 exe 확장자를 갖는 앱 설치 파일 등도 바이너리 파일이다.
# 
# 반면에 텍스트 파일은 내용을 확인하기 위해 특별한 기능이 필요 없고
# 임의의 텍스트 편집기를 사용해도 내용을 바로 확인할 수 있다.
# 여기서는 텍스트 파일만을 대상으로 파이썬 코드를 이용하여 
# 파일 다운로드, 열기, 수정하기 등을 살펴본다.

# **디렉토리와 폴더**
# 
# 디렉토리<font size='2'>directory</font>와 폴더<font size='2'>folder</font>는 동일한 개념이다. 
# 다만 리눅스 계열 운영체제에서는 디렉토리를, 윈도우 운영체제에서는 폴더를 선호한다.
# 그리고 리눅스 계열 운영체제에서는 디렉토리를 파일이라고 부르기도 한다.
# 하지만 여기서는 디렉토리를 폴더 개념으로만 사용하며 파일과는 구분한다.

# **경로**
# 
# 경로는 특정 디렉토리 또는 파일의 위치를 나타내는 문자열이다. 
# 사용하는 운영체제마다 표현 방법이 다르기에 조심해야 한다.
# 예를 들어 현재 파이썬 코드가 실행되는 디렉토리의 경로는 
# 리눅스와 윈도우의 경우 다음과 같이 다르게 표현된다.
# 
# - 리눅스의 경우: '/home/gslee/Documents/GitHub/datapy/jupyter-book'
# - 윈도우의 경우: 'C:\Users\gslee\Documents\GitHub\datapy\jupyter-book'
# 
# 하지만 `pathlib` 모듈의 `Path` 클래스를 이용하면 운영체제를 신경쓰지 않고 경로를 다룰 수 있다.

# **현재 작업 디렉토리**
# 
# 보통 `cwd` 라고 줄여서 사용되는 **현재 작업 디렉토리**<font size='2'>current working directory</font>는
# 현재 파이썬이 실행되는 디렉토리 경로를 가리킨다. 
# 현재 작업 디렉토리는 `os` 모듈의 `getcwd()` 함수를 이용하여 확인할 수 있다.
# `os` 모듈은 운영체제를 조작하는 API를 많이 갖고 있다.

# In[1]:


import os

print(os.getcwd())


# **절대경로와 상대경로**
# 
# 앞서 언급한 두 개의 경로는 
# 운영체제의 맨 상위 디렉토리를 기준으로 하는 경로라는 의미에서 **절대경로**라 부른다.
# 반면에 **상대경로**는 현재 작업 디렉토리(cwd)를 기준으로 경로를 작성한다. 
# 만약에 cwd가 `Documents` 라면, 위 두 개의 경로의 상대경로는 다음과 같다.
# 
# - 리눅스의 경우: 'GitHub/pybook/jupyter-book'
# - 윈도우의 경우: 'GitHub\pybook\jupyter-book'
# 
# 아래와 같이 사용할 수 있다. 점(`.`)이 현재 작업 디렉토리를 가리킨다.
# 
# - 리눅스의 경우: './GitHub/pybook/jupyter-book'
# - 윈도우의 경우: '.\GitHub\pybook\jupyter-book'

# ## 파일 다운로드

# `segismuldo.txt` 파일을 인터넷에서 다운로드 하여 `examples` 라는 하위 폴더에 저장하려 한다.
# 이를 위해 먼저 해당 디렉토리를 생성해야 한다.

# **`pathlib.Path` 클래스: 디렉토리 지정**
# 
# 파일을 저장할 디렉토리를 지정한다.
# 이를 위해 `pathlib` 모듈의 `Path` 클래스를 이용한다.
# `Path` 클래스는 지정된 디렉토리 또는 파일의 경로를 담은 객체를 생성한다.
# 
# - `Path()` : 현재 작업 디렉토리를 나타내는 객체. `Path('.')` 과 동일한 의미임.
# - 슬래시 연산자 `/`: 두 개의 경로를 이어붙히는 연산자. 왼쪽 인자는 `Path` 객체, 둘째 인자는 문자열.

# 아래 코드의 `data_path` 변수는 현재 작업 디렉토리에 위치한 `examples` 라는 폴더의 경로를 가리킨다.

# In[2]:


from pathlib import Path

data_path = Path() / "examples"


# **`Path` 객체**

# `Path` 객체는 다양한 정보를 다루는 메서드와 속성을 제공한다.
# 예를 들어, 현재 작업 디렉토리의 경로를 확인하려면 `cwd()` 메서드를 실행한다.

# In[3]:


data_path.cwd()


# `name` 속성은 경로가 가리키는 디렉토리 또는 파일 이름을 저장한다. 

# In[4]:


data_path.name


# `parent` 속성은 지정되 경로가 가리키는 디렉토리 또는 파일이 저장된 부모 디렉토리의 이름을 저장한다.
# `data_path` 가 현재 디렉토리의 하위 디렉토리인 `data`를 가리키기에
# 그것의 부모 디렉토리인 현재 디렉토리를 가리키는 점(`'.'`) 이 저장된다.

# In[5]:


data_path.parent


# **`Path.mkdir()` 메서드**
# 
# `Path` 객체의 `mkdir()` 메서드를 이용하여 지정된 경로에 해당하는 폴더를 실제로 생성한다. 
# 다음 두 개의 옵션 인자를 사용한다.
# 
# - `parent=True`: 부모 디렉토리가 필요하면 생성할 것.
# - `exist_ok = True`: 디렉토리 이미 존재하면 그대로 사용할 것. 

# In[6]:


data_path.mkdir(parents=True, exist_ok=True)


# **`urllib.request` 모듈**

# 아래 코드는 인터넷 특정 사이트에 저장되어 있는 `segismundo.txt` 파일을 다운로드 한다.
# 
# - 사이트 주소: `https://raw.githubusercontent.com/codingalzi/pydata/master/notebooks/examples/`
# - 파일 이름: `segismundo.txt`

# `urllib.request` 모듈은 인터넷 사이트를 다루는 다양한 API를 제공한다.
# 여기서는 `urlretrieve()` 함수를 이용하여 파일을 다운로드하여 지정된 폴더에 지정된 이름으로 저장한다.
# 
# - 첫째 인자: 파일 링크 문자열
# - 둘째 인자: 파일 저장 위치 경로(파일명 포함) 문자열
# - 반환값: 저장된 파일의 경로와 파일 정보. 파일 정보는 중요하지 않음.

# In[7]:


import urllib.request

# 사이트 주소
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/codingalzi/pydata/master/notebooks/examples/"
# 파일명
filename = "segismundo.txt"
# 다운로드 파일 링크
url = DOWNLOAD_ROOT + filename
# 다운로드 실행
urllib.request.urlretrieve(url, "examples/" + filename)


# ##  파일 열기

# ### 파일 불러오기

# 파일로 저장된 텍스트 문서를 파일 데이터로 불러오려면 `open()` 함수를 사용한다.
# 문서의 내용을 담은 파일 객체가 반환된다.

# In[8]:


path = 'examples/segismundo.txt'
f = open(path)


# 반환값의 자료형은 `_io.TextIOWrapper` 라는 특별한 객체이며, 굳이 자세히 알 필요 없다.

# In[9]:


type(f)


# ###  파일 데이터 내용 읽기

# 파일 객체의 내용은 `for` 반복문을 이용하여 한 줄씩 확인할 수 있다.
# 그런데 `segismundo.txt` 파일엔 스페인어로 작성된 시가 저장되어 있으며 &#xF1;, &#xE1; 등의
# 특수 문자가 사용된다.
# 따라서 한글 윈도우 운영체제에서는 아래와 같이 일부 표현이 깨질 수 있다.

# ```python
# for line in f:
#     print(line)
# 
# Sue챰a el rico en su riqueza,
# 
# que m찼s cuidados le ofrece;
# 
# 
# 
# sue챰a el pobre que padece
# 
# su miseria y su pobreza;
# 
# 
# 
# sue챰a el que a medrar empieza,
# 
# sue챰a el que afana y pretende,
# 
# sue챰a el que agravia y ofende,
# 
# 
# 
# y en el mundo, en conclusi처n,
# 
# todos sue챰an lo que son,
# 
# aunque ninguno lo entiende.
# ```

# 이유는 한글 윈도우 운영체제가 파일을 열거나 생성할 때 기본적으로 `cp949` 인코딩 방식을 사용하는 반면에
# 스페인어 등에 사용되는 특수 문자는 기본적으로 `utf-8` 방식으로 인코딩되기 때문이다.
# 따라서 아래와 같이 인코딩 방식을 utf-8로 지정하여 파일을 열거나 생성하면 운영체제와 상관 없이 
# 모든 유니코드 특수 문자를 제대로 처리한다.

# In[10]:


# 파일 경로를 변수에 할당하면 편리함
path = 'examples/segismundo.txt'

f = open(path, encoding='utf-8')

for line in f:
    print(line)


# 한 번 내용을 들여다 본 파일 객체의 내용은 더 이상 확인할 수 없음에 주의하라.

# In[11]:


for line in f:
    print(line)


# 이유는 파일 객체가 이터레이터라는 특수한 객체이기 때문인데 여기서는 자세히 설명하지 않는다.
# 암튼, 파일 내용을 다시 확인하려면 `open()` 함수를 실행해야 한다는 점만 기억한다.

# **`strip()` 문자열 메서드 활용**

# 줄 간격이 너무 큰 것으로 보아 줄 끝에 줄바꿈 문자(`\n`)가 포함된 것으로 보인다.
# 이와 더불어 `print()` 함수는 기본적으로 줄바꿈을 넣어주기에 줄바꿈을 두 번하게 된다.
# 이런 경우 `strip()` 문자열 메서드를 이용하여 문자열 양 끝에 위치한 스페이스, 탭, 줄바꿈 등의 공백 기호를 제거하는 게 좋다. 

# In[12]:


f = open(path, encoding='utf-8')

for line in f:
    print(line.strip())


# 아래 코드는 파일 내용을 단순히 출력하지 않고, 리스트로 저장한다.
# 이를 위해 리스트 조건제시법을 이용한다.

# In[13]:


f = open(path, encoding='utf-8')

lines = [x.strip() for x in f]
lines


# ### 파일 닫기

# **`close()` 파일 객체 메서드**

# 저장되어 있는 파일 내용을 불러오고 더 이상 필요 없으면 해당 파일 자체를 
# `close()` 메서드를 호출하여 닫아야 한다.
# 그렇지 않으면 마치 편집기에 해당 파일이 열려 있는 것처럼 의도치 않은 수정이 파일에 가해질 수 있다.

# ```python
# f = open(path, encoding='utf-8')
# lines = [x.strip() for x in f]
# f.close()  # 파일이 더 이상 열려 있을 필요가 없는 경우
# ```

# **`with ... as ...` 형식**

# 파일을 불러와 일을 처리한 다음에 파일닫기를 자동으로 진행하게 하려면 
# 아래에서 처럼 `with ... as ...` 형식을 사용한다. 
# 아래 코드는 `f.close()` 메서드를 자동으로 마지막에 실행해서 열린 파일을 닫는다.

# In[14]:


with open(path, encoding='utf-8') as f:
    lines = [x.strip() for x in f]


# In[15]:


lines


# 닫은 파일은 더 이상 내용을 들여다 볼 수 없다.

# ```python
# for line in f:
#     print(line)
#     
# ---------------------------------------------------------------------------
# ValueError                                Traceback (most recent call last)
# c:\Users\gslee\Documents\GitHub\datapy\jupyter-book\python_basic_7.ipynb Cell 38 in <cell line: 1>()
# ----> 1 for line in f:
#       2     print(line)
# 
# ValueError: I/O operation on closed file.
# ```

# ### `readline()`와 `readlines()` 파일 객체 메서드 

# 열려 있는 파일이 이터레이터이기에 `for` 반복문으로 모든 줄을 순회하하면서 내용을 확인할 수 있었다.
# 반면에 아래 두 메서드를 이용하면 `for` 반복문 없이 줄 단위로 내용을 확인할 수 있다.
# 
# | 메서드 | 기능 |
# |---|---|
# | `readline()` | 한 줄씩 읽어서 반환한다. 이터레이터의 `__next__()` 메서드 역할을 한다.|
# | `readlines()` | 모든 줄을 한꺼번에 읽어와서 각각의 줄을 항목으로 갖는 문자열의 리스스를 반환한다. |

# In[16]:


f = open(path, encoding='utf-8')


# `readline()` 메서드를 실행할 때마다 한 줄씩 읽어준다.

# In[17]:


f.readline()


# In[18]:


f.readline()


# In[19]:


f.readline()


# In[20]:


f.close()


# 반면에 `readlines()` 메서드는 문자열의 리스트를 반환한다.
# 항목은 줄 단위로 가져온 문자열이다.

# In[21]:


f = open(path, encoding='utf-8')

f.readlines()


# __참고:__ `readlines()`의 결과를 보면 각 줄의 끝에 줄바꿈 기호가 사용되었음을 볼 수 있다.

# ## 파일 생성 및 작성하기

# 아래 두 가지 방식으로 파일을 생성하거나 작성할 수 있다.
# 
# * 방식 1: `open('파일명', 'w')`
#     * 파일이 이미 존재할 경우 내용을 완전히 지우고 빈 파일로 불러옴
# * 방식 2: `open('파일명', 'x')`
#     * 파일이 이미 존재할 경우 오류 발생

# 아래 코드는 'segismundo.txt'의 파일 내용을 읽어들이자 마자 바로 'tmp.txt' 파일에 써버린다.
# 단, 빈줄은 무시한다. 
# 
# | 메서드 | 기능 |
# |---|---|
# |`writelines()` | 문자열들의 리스트, 튜플에 포함된 항목을 모두 파일에 순서대로 저장 |
# | `write()` | 지정된 문자열을 파일 맨 아랫줄에 추가 |

# __참고:__ `open()` 함수가 반환하는 파일 객체를 __파일 핸들__(file handle) 이라 부른다. 따라서 
# 아래 코드에서처럼 파일 객체를 가리키는 변수로 handle, file_handle 등을 많이 사용한다.

# In[22]:


with open('tmp.txt', 'w', encoding="utf-8") as handle:
    handle.writelines([x for x in open(path, encoding="utf-8") if len(x) > 1])
    handle.write("끝에서 둘째줄입니다.")
    handle.write("마지막 줄입니다.")


# 내용을 다시 확인해보자.

# In[23]:


with open('tmp.txt', encoding='utf-8') as handle:
    lines = handle.readlines()
    
lines


# 그런데 마지막 두 줄이 구분되지 않는다. 이유는 끝에서 두번째 줄 끝에 줄바꿈 문자가 포함되지 않았기 때문이다.
# 아래처럼 하면 줄바꿈이 이루어진다.

# In[24]:


with open('tmp.txt', 'w', encoding="utf-8") as handle:
    handle.writelines([x for x in open(path, encoding="utf-8") if len(x) > 1])
    handle.write("끝에서 둘째줄입니다.\n")
    handle.write("마지막 줄입니다.")


# In[25]:


with open('tmp.txt', encoding='utf-8') as handle:
    lines = handle.readlines()
    
lines

