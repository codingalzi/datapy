#!/usr/bin/env python
# coding: utf-8

# (ch:pyplot2)=
# # matplotlib.pyplot 2부

# **참고**
# 
# 웨스 맥키니의 [<파이썬 라이브러리를 활용한 데이터 분석>](https://github.com/wesm/pydata-book)의 
# 9장1절에 사용된 소스코드의 일부를 활용한다.

# **기본 설정**

# In[1]:


import numpy as np


# 맷플롯립<font size='2'>Matplotlib</font>은 간단한 그래프 도구를 제공하는 라이브러리다. 
# 맷플롯립의 대부분의 함수는 파이플롯<font size='2'>pyplot</font> 모듈에 포함되어 있으며
# 관행적으로 `plt` 별칭으로 불러온다.

# In[2]:


import matplotlib.pyplot as plt


# ## `Figure` 객체와 서브플롯(subplot)

# 모든 그래프는 `Figure` 객체 내에 존재하며, `plt.figure()` 함수에 의해 생성된다.
# 
# ```python
# fig = plt.figure()
# ```

# `Figure` 객체 내에 그래프를 그리려면 서브플롯(subplot)을 지정해야 한다.
# 아래 코드는 `add_subplot()` 함수를 이용하여 지정된 `Figure` 객체안에 그래프를 그릴 공간을 준비한다.
# (nrows, ncols, index) 형식의 인자의 의미는 다음과 같다.
# 
# - nrows: 이미지 행의 개수
# - ncols: 이미지 칸의 개수
# - index: 이미지의 인덱스. 1부터 시작.

# 만약 2x2 모양으로 총 4개의 이미지를 생성하려면 아래와 같이 실행한다.

# ```python
# fig = plt.figure()
# 
# ax1 = fig.add_subplot(2, 2, 1)
# ax2 = fig.add_subplot(2, 2, 2)
# ax3 = fig.add_subplot(2, 2, 3)
# ax4 = fig.add_subplot(2, 2, 4)
# ```

# **그래프 삽입하기**

# 두 가지 방식으로 각각의 서브플롯에 그림을 삽입할 수 있다.

# *방식 1: `matplotlib.pyplot.plot()` 함수 활용*

# 이 방식은 항상 마지막에 선언된 서브플롯에 그래프를 그린다.
# 예를 들어 아래 코드는 무작위로 선택된 50개의 정수들의 누적합으로 이루어진 데이터를
# 파선 그래프로 그린다.

# In[3]:


np.random.seed(12345)

data1 = np.random.randn(50).cumsum()
data2 = np.random.randn(50).cumsum()


# * 2x1 모양의 2개 이미지 사용하기

# In[4]:


fig = plt.figure()

ax1 = fig.add_subplot(2, 1, 1)
plt.plot(data1, 'k--')

ax2 = fig.add_subplot(2, 1, 2)
plt.plot(data2, 'k--')


# * 1x2 모양의 2개 이미지 사용하기

# In[5]:


fig = plt.figure()

ax1 = fig.add_subplot(1, 2, 1)
plt.plot(data1, 'k--')

ax2 = fig.add_subplot(1, 2, 2)
plt.plot(data2, 'k--')


# *방식 2: `객체명.plot()` 함수 활용*

# 특정 서브플롯에 그래프를 삽입하려면 서브플롯 객체의 이름과 함께 `plot()` 함수 등을 호출해야 한다.

# In[6]:


np.random.seed(12345)

fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)

# 위치: (2, 2, 1)
ax1.hist(np.random.randn(100), bins=20, color='k', alpha=.1)
# 위치: (2, 2, 2)
ax2.scatter(np.arange(30), np.arange(30) + 3 * np.random.randn(30))
# 위치: (2, 2, 3)
ax3.plot(np.random.randn(50).cumsum(), 'k--')
# 위치: (2, 2, 4)
ax4.plot(np.random.randn(50).cumsum(), 'k--')

plt.show()


# `Figure` 객체의 크기는 `plt.rc()` 함수를 이용해서 아래와 같이 지정한다.
# `plt.rc()` 함수의 자세한 활용법은 잠시 뒤에 보다 자세히 살펴 본다.

# In[7]:


plt.rc('figure', figsize=(10, 6))


# 이제 네 개의 이미지가 보다 적절한 크기로 그려진다.

# In[8]:


np.random.seed(12345)

fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)

# 위치: (2, 2, 1)
ax1.hist(np.random.randn(100), bins=20, color='k', alpha=.1)
# 위치: (2, 2, 2)
ax2.scatter(np.arange(30), np.arange(30) + 3 * np.random.randn(30))
# 위치: (2, 2, 3)
ax3.plot(np.random.randn(50).cumsum(), 'k--')
# 위치: (2, 2, 4)
ax4.plot(np.random.randn(50).cumsum(), 'k--')

plt.show()


# **서브플롯 관리**

# `matplotlib.pyplot.subplots()` 함수는 여러 개의 서브플롯을 포함하는 `Figure` 객체를 관리해준다.
# 예를 들어, 아래 코드는 2x3 크기의 서브플롯을 담은 (2,3) 모양의 넘파이 어레이로 생성한다.
# 
# * 반환값: `Figure` 객체와 지정된 크기의 넘파이 어레이. 각 항목은 서브플롯 객체임.

# In[9]:


fig, axes = plt.subplots(2, 3)
axes


# x 축 또는 y 축 눈금을 공유할 수 있다.

# In[10]:


fig, axes = plt.subplots(2, 3, sharex=True, sharey=True)
axes


# `plt.subplots_adjust()` 함수는 각 서브플롯 사이의 여백을 조절하는 방식을 보여준다. 
# 여백의 크기는 그래프의 크기와 숫자에 의존한다.

# * 여백이 0일 때:

# In[11]:


fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)

for i in range(2):
    for j in range(2):
        axes[i, j].plot(np.random.randn(50))

plt.subplots_adjust(wspace=0, hspace=0) # 상하좌우 여백: 0


# * 여백이 0.1일 때:

# In[12]:


fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)

for i in range(2):
    for j in range(2):
        axes[i, j].plot(np.random.randn(50))

plt.subplots_adjust(wspace=0.1, hspace=0.1) # 상하좌우 여백: 0.1


# ## 눈금과 라벨

# **이미지 타이틀, 축 이름, 눈금, 눈금 이름 지정**

# 두 가지 방식으로 진행할 수 있다.

# *방식 1: 파이플롯 객체의 메서드 활용*

# - `set_xticks()` 함수: 눈금 지정
# - `set_xticklabels()` 함수: 눈금 라벨 지정
# - `set_title()` 함수: 그래프 타이틀 지정
# - `set_xlabel()` 함수: x축 이름 지정

# In[13]:


data = np.random.randn(100).cumsum()


# In[14]:


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(data)

ticks = ax.set_xticks([0, 25, 50, 75, 100])
labels = ax.set_xticklabels(['one', 'two', 'three', 'four', 'five'])

ax.set_title('Random Plot')
ax.set_xlabel('Levels')


# 눈금 크기와 방향도 지정할 수 있다.

# In[15]:


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(data)

ticks = ax.set_xticks([0, 25, 50, 75, 100])
labels = ax.set_xticklabels(['one', 'two', 'three', 'four', 'five'],
                            rotation=90, fontsize='small')

ax.set_title('Random Plot')
ax.set_xlabel('Levels')


# *방식 2: pyplot 모듈의 함수 활용*

# 이 방식은 마지막에 선언된 서브플롯에 대해서만 작동한다.
# 
# - `plt.xticks()` 함수: 눈금 및 눈금 라벨 지정
# - `plt.title()` 함수: 그래프 타이틀 지정
# - `plt.xlabel()` 함수: x축 이름 지정

# In[16]:


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(data)

plt.xticks([0, 25, 50, 75, 100], ['one', 'two', 'three', 'four', 'five'],
            rotation=90, fontsize='small')

ax.set_title('Random Plot')
ax.set_xlabel('Levels')


# ## 주석

# 이미지에 주석을 추가할 수 있다.
# 
# 설명을 위해 S&P 500 (스탠다드 앤 푸어스, Standard and Poor's 500)의 미국 500대 기업을 포함한 
# 주식시장지수 데이터로 그래프를 생성하고 2007-2008년 사이에 있었던 
# 재정위기와 관련된 중요한 날짜를 주석으로 추가한다.

# In[17]:


import pandas as pd

spx_path = 'https://raw.githubusercontent.com/codingalzi/datapy/master/jupyter-book/examples/spx.csv'


# In[18]:


data = pd.read_csv(spx_path)
data


# 여기서는 시간 컬럼을 행의 인덱스로 사용한다.
# 
# - `index_col=0`: 0번 열(column)을 인덱스로 사용
# - `parse_dates=True`: 년월일까지만 구분해서 인덱스로 사용하도록 함. 기본값은 `False`.

# In[19]:


data = pd.read_csv(spx_path, index_col=0, parse_dates=True)
data


# 하나의 열만 존재하는 데이터프레임이기에 시리즈로 변환한다.
# 
# __참고:__ 반드시 필요한 과정은 아니다. `spx` 대신 `data`를 그대로 사용해도 동일하게 작동한다.

# In[20]:


spx = data['SPX']
spx


# 위 데이터를 단순하게 그래프로 나타내면 다음과 같다.

# In[21]:


from datetime import datetime

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.plot(spx, 'k-')


# 2007-2008년 세계적 금융위기 지점을 아래 내용으로 그래프에 주석으로 추가해보자.
# 
# - 2007년 10월 11일: 주가 강세장 위치
# - 2008년 3월 12일: 베어스턴스 투자은행 붕괴
# - 2008년 9월 15일: 레만 투자은행 파산

# In[22]:


crisis_data = [
    (datetime(2007, 10, 11), 'Peak of bull market'),
    (datetime(2008, 3, 12), 'Bear Stearns Fails'),
    (datetime(2008, 9, 15), 'Lehman Bankruptcy')
]


# **`annotate()` 메서드 활용**

# - `xt` 속성: 화살표 머리 위치
# - `xytext` 속성: 텍스트 위치
# - `arrowprops` 속성: 화살표 속성
# - `horizontalalignment`: 텍스트 좌우 줄맞춤
# - `verticalalignment`: 텍스트 상하 줄맞춤

# In[23]:


from datetime import datetime

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

spx.plot(ax=ax, style='k-')

for date, label in crisis_data:
    ax.annotate(label, 
                xy=(date, spx.asof(date) + 75),
                xytext=(date, spx.asof(date) + 225),
                arrowprops=dict(facecolor='red', headwidth=4, width=2,
                                headlength=4),
                horizontalalignment='left', verticalalignment='top')

# 2007-2010 사이로 확대
ax.set_xlim(['1/1/2007', '1/1/2011'])
ax.set_ylim([600, 1800])

ax.set_title('Important dates in the 2008-2009 financial crisis')


# ## matplotlib 기본 설정

# `plt.rc()` 함수를 이용하여 matplot을 이용하여 생성되는 이미지 관련 설정을 전역적으로 지정할 수 있다.
# 사용되는 형식은 다음과 같다.
# 
# - 첫째 인자: 속성 지정
# - 둘째 인자: 속성값 지정
# 
# __참고:__ 'rc' 는 기본설정을 의미하는 단어로 많이 사용된다. 
# 풀어 쓰면 "Run at startup and they Configure your stuff", 
# 즉, "프로그램이 시작할 때 기본값들을 설정한다"의 의미이다.
# '.vimrc', '.bashrc', '.zshrc' 등 많은 애플리케이션의 초기설정 파일명에 사용되곤 한다.
# 
# 아래 코드는 이미지의 사이즈를 지정한다.

# In[24]:


plt.rc('figure', figsize=(6, 6))


# In[25]:


from datetime import datetime

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

spx.plot(ax=ax, style='k-')

for date, label in crisis_data:
    ax.annotate(label, 
                xy=(date, spx.asof(date) + 75),
                xytext=(date, spx.asof(date) + 225),
                arrowprops=dict(facecolor='black', headwidth=4, width=2,
                                headlength=4),
                horizontalalignment='left', verticalalignment='top')

# 2007-2010 사이로 확대
ax.set_xlim(['1/1/2007', '1/1/2011'])
ax.set_ylim([600, 1800])

ax.set_title('Important dates in the 2008-2009 financial crisis')


# 아래 코드는 다양한 속성을 지정하는 방식을 보여준다.

# * 이미지 사이즈 지정

# In[26]:


plt.rc('figure', figsize=(10, 6))


# * 선 속성 지정

# In[27]:


plt.rc('lines', linewidth=3, color='b')


# * 텍스트 폰트 속성 지정

# In[28]:


font_options = {'family' : 'monospace',
                'weight' : 'bold',
                'size'   : '15'}
plt.rc('font', **font_options)


# * 그래프 구성 요소의 색상 지정

# In[29]:


plt.rcParams['text.color'] = 'blue'
plt.rcParams['axes.labelcolor'] = 'red'
plt.rcParams['xtick.color'] = 'green'
plt.rcParams['ytick.color'] = '#CD5C5C'  # RGB 색상


# 아래 코드는 앞서 설정된 다양한 속성을 반영한 결과를 보여준다.

# In[30]:


from datetime import datetime

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

# spx.plot(ax=ax, style='k-')
spx.plot(ax=ax, style='-')    # 기본 색상 사용

for date, label in crisis_data:
    ax.annotate(label, 
                xy=(date, spx.asof(date) + 75),
                xytext=(date, spx.asof(date) + 225),
                arrowprops=dict(facecolor='black', headwidth=4, width=2,
                                headlength=4),
                horizontalalignment='left', verticalalignment='top')

# 2007-2010 사이로 확대
ax.set_xlim(['1/1/2007', '1/1/2011'])
ax.set_ylim([600, 1800])

ax.set_title('Important dates in the 2008-2009 financial crisis')


# ## 연습문제

# 참고: [(실습) matplotlib.pyplot 2부](https://colab.research.google.com/github/codingalzi/datapy/blob/master/practices/practice-data_visualization_pyplot2.ipynb)
