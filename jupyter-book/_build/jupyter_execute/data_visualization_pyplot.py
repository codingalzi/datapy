#!/usr/bin/env python
# coding: utf-8

# # 데이터 시각화: matplotlib.pyplot

# **참고**
# 
# [Matplotlib Tutorial](https://www.w3schools.com/python/matplotlib_intro.asp) 를 참고하였습니다. 
# 
# - annotation 추가 필요

# **기본 설정**

# In[1]:


import numpy as np


# 맷플롯립<font size='2'>Matplotlib</font>은 간단한 그래프 도구를 제공하는 라이브러리다. 

# In[2]:


import matplotlib


# 설치된 버전은 다음과 같이 확인한다.

# In[3]:


matplotlib.__version__


# 맷플롯립의 대부분의 함수는 파이플롯<font size='2'>pyplot</font> 모듈에 포함되어 있으며
# 관행적으로 `plt` 별칭으로 불러온다.

# In[4]:


import matplotlib.pyplot as plt


# ## `plot()` 함수

# **선분 그리기**

# 2차원 평면의 두 점이 주어졌을 때 두 점을 잇는 선분을 그린다. 
# 두 점의 좌표는 x 축 기준으로 두 개의 값을,
# y 축 기준으로 두 개의 값을 지정하면
# 순서대로 쌍을 지어 사용된다.
# 
# 예를 들어 (0, 0), (2, 100), (4, 60), (5, 200) 네 점을 잇는 선분을 그리려면
# 다음처럼 `[0, 2, 4, 6]` 를 x 좌표들의 어레이로, 
# `[0, 100, 60, 200]` 을 y 자표들의 어레이로 선언하고
# `plot()` 함수의 두 인자로 지정한다.

# In[5]:


xpoints = np.array([0, 2, 4, 6])
ypoints = np.array([0, 120, 60, 200])

plt.plot(xpoints, ypoints)
plt.show()


# 선분 색깔을 빨강으로 바꾸려면 셋째 인자로 `'r'` 을 사용한다.

# In[6]:


xpoints = np.array([0, 2, 4, 6])
ypoints = np.array([0, 120, 60, 200])

plt.plot(xpoints, ypoints, 'r')
plt.show()


# **점만 그리기**

# 두 점을 잇는 선분을 그리지 않으려면 셋째 인자로 `'o'` 를 지정한다.

# In[7]:


xpoints = np.array([0, 2, 4, 6])
ypoints = np.array([0, 120, 60, 200])

plt.plot(xpoints, ypoints, 'o')
plt.show()


# 빨간 덧셈 기호로 표기하고 싶으면 `r+` 를 셋째 인자로 사용한다.

# In[8]:


xpoints = np.array([0, 2, 4, 6])
ypoints = np.array([0, 120, 60, 200])

plt.plot(xpoints, ypoints, '+r')
plt.show()


# **x 좌표 생략하기**

# x 좌표를 생략할 수 있다.
# 그러면 y 좌표에 사용된 개수만큼의 인덱스가 자동으로 x 좌표로 사용된다.

# In[9]:


ypoints = np.array([5, 1, 2, 10, 7, 9])

plt.plot(ypoints)
plt.show()


# ## 마커

# 각 점을 강조하기 위해 모양을 지정하고 싶을 때 `marker` 옵션 인자를 사용한다. 
# 예를 들어, 작은 원으로 각 점을 표시하고 싶으면 `market='o'` 로 지정한다.

# In[10]:


ypoints = np.array([5, 1, 2, 10, 7, 9])

plt.plot(ypoints, marker='o')
plt.show()


# 별표를 사용하려면 `marker='*'` 를 사용한다.

# In[11]:


ypoints = np.array([5, 1, 2, 10, 7, 9])

plt.plot(ypoints, marker='*')
plt.show()


# **마커 종류**

# [다양한 종류의 마커<font size='2'>marker</font>](https://matplotlib.org/stable/api/markers_api.html)가 
# 제공된다. 
# 그중 일부는 다음과 같다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/datapy/master/jupyter-book/images/pyplot-marker.png" width="40%"></div>
# 
# 출처: [Matplotlib](https://matplotlib.org/stable/api/markers_api.html)

# **포맷 문자열**

# 점 모양, 선분 사용 여부 등을 한꺼번에 지정할 수 있다.
# 예를 들어, 빨간 점을 잇는 선분을 사용하려면 셋째 인자로 `o-r` 을 사용한다.

# In[12]:


ypoints = np.array([5, 1, 2, 10, 7, 9])

plt.plot(ypoints, 'o-r')
plt.show()


# 초록 점선을 사용하려면 `o:g` 를 셋째 인자로 사용한다.

# In[13]:


ypoints = np.array([5, 1, 2, 10, 7, 9])

plt.plot(ypoints, 'o:g')
plt.show()


# 포맷 문자열은 세 개의 요소로 구성되며, 각 요소는 생략될 수 있다.
# 
# - 점 모양
# - 선분 모양
# - 색

# 점 모양은 앞서 소개한 마커<font size='2'>marker</font>에 의해 정해진다.
# 지정되지 않으면 점이 표기되지 않는다.

# **선분 모양**

# 선 모양은 다음 중 하나를 선택한다.
# 지정되지 않으면 선이 그려지지 않는다.
# 
# | 선 기호 | 선 모양 |
# | :--- | :--- |
# | '-' | 실선 |
# | ':' | 점선 |
# | '--' | 파선 |
# | '-.' | 1점 쇄선 |

# In[14]:


ypoints = np.array([5, 1, 2, 10, 7, 9])

plt.plot(ypoints, 'ok')
plt.show()


# 1점 쇄선은 아래 모양이다.

# In[15]:


ypoints = np.array([5, 1, 2, 10, 7, 9])

plt.plot(ypoints, 'o-.g')
plt.show()


# **색 선택**

# 색은 다음 중 하나를 선택한다.
# 생략되면 파랑이 기본값으로 사용된다.
# 
# | 색 기호 | 색 |
# | :--- | :--- |
# | 'r' | 빨강 |
# | 'g' | 초록 |
# | 'b' | 파랑 |
# | 'c' | 청록 |
# | 'm' | 심홍 |
# | 'y' | 노랑 |
# | 'k' | 검정 |
# | 'w' | 하양 |

# **마커 크기**

# 마커의 크기는 `markersize` 또는 `ms` 옵션 인자로 지정한다.

# In[16]:


ypoints = np.array([5, 1, 2, 10, 7, 9])

plt.plot(ypoints, 'o--g', ms=20)
plt.show()


# **마커 테두리 색**

# 마커의 테두리 색을 `markeredgecolor` 또는 `mec` 옵션 인자로 지정한다.

# In[17]:


ypoints = np.array([5, 1, 2, 10, 7, 9])

plt.plot(ypoints, 'o--g', ms=20, mec='r')
plt.show()


# **마커 내부 색**

# 마커의 내부의 색을 `markerfacecolor` 또는 `mfc` 옵션 인자로 지정한다.

# In[18]:


ypoints = np.array([5, 1, 2, 10, 7, 9])

plt.plot(ypoints, 'o--g', ms=20, mfc='r')
plt.show()


# **마커 색**

# `mec` 와 `mfc` 인자를 동일한 값으로 지정하면 마커의 색을 선분의 색과 다르게 지정할 수 있다.

# In[19]:


ypoints = np.array([5, 1, 2, 10, 7, 9])

plt.plot(ypoints, 'o--g', ms=20, mfc='r', mec='r')
plt.show()


# **헥스 코드 활용**

# 색을 지정하기 위해 [헥스 코드](https://www.w3schools.com/colors/colors_hexadecimal.asp)를
# 사용할 수 있다.

# In[20]:


ypoints = np.array([5, 1, 2, 10, 7, 9])

plt.plot(ypoints, 'o-.g', ms=20, mfc='#ab2b54', mec='#ab2b54')
plt.show()


# **색 이름 활용**

# [140개의 색 이름](https://www.w3schools.com/colors/colors_names.asp)을
# 직접 사용할 수 있다.

# In[21]:


ypoints = np.array([5, 1, 2, 10, 7, 9])

plt.plot(ypoints, 'o:g', ms=20, mfc='blueviolet', mec='blueviolet')
plt.show()


# ## 선분

# `linestyle` 또는 `ls` 옵션 인자로 선 모양을 지정할 수 있다.
# 앞서 언급한 선 기호를 사용하거나 다음 문자열 중에 하나를 사용한다.
# 
# 

# | 선 기호 | 문자열 |
# | :--- | :--- |
# | '-' | 'solid' |
# | ':' | 'dotted' |
# | '--' | 'dashed' |
# | '-.' | 'dashdot |

# In[22]:


ypoints = np.array([5, 1, 2, 10, 7, 9])

plt.plot(ypoints, ls='dotted')
plt.show()


# **선 색깔**

# `color` 또는 `c` 옵션 인자로 선의 색을 지정하며,
# 인자로 사용되는 값은 앞서 마커의 색에서 설명한 방식과 동일하게 지정한다.

# **선 두께**

# `linewidth` 또는 `lw` 옵션 인자로 선의 두께를 지정한다.

# In[23]:


ypoints = np.array([5, 1, 2, 10, 7, 9])

plt.plot(ypoints, ls='solid', lw='15')
plt.show()


# **여러 개의 선 그리기**

# `plt.plot()` 함수를 여러 번 호출하면 여러 개의 선이 그려진다.
# 단. `plt.show()` 함수를 항상 맨 나중에 호출해야 한다.
# 선들을 구분하기 위해 자동으로 다른 색이 사용된다.
# 물론 선의 색을 따로따로 지정할 수도 있다.

# In[24]:


y1 = np.array([5, 1, 2, 10, 7, 9])
y2 = np.array([3, 7, 4, 2, 9, 6])

plt.plot(y1)
plt.plot(y2)

plt.show()


# 두 개 이상의 선을 함께 그릴 수도 있다.

# In[25]:


y1 = np.array([5, 1, 2, 10, 7, 9])
y2 = np.array([3, 7, 4, 2, 9, 6])
y3 = np.array([4, 6, 8, 6, 4, 5])

plt.plot(y1)
plt.plot(y2)
plt.plot(y3)

plt.show()


# 각 선의 x 좌표가 다르다면 아래 방식으로 지정한다.

# In[26]:


x1 = np.array([0, 1, 3, 5, 7, 10])
y1 = np.array([5, 1, 2, 10, 7, 9])
x2 = np.array([0, 2, 4, 6, 8, 10])
y2 = np.array([3, 7, 4, 2, 9, 6])

plt.plot(x1, y1, x2, y2)

plt.show()


# ## 레이블과 제목
