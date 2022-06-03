#!/usr/bin/env python
# coding: utf-8

# (ch:pyplot2)=
# # matplotlib.pyplot 2부

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

