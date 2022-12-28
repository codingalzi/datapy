#!/usr/bin/env python
# coding: utf-8

# # 소개

# [&lt;파이썬 라이브러리를 활용한 데이터 분석(2판)&gt;의 소스코드](https://github.com/wesm/pydata-book)를 
# 담고 있는 주피터 노트북과
# [&lt;밑바닥부터 시작하는 데이터 과학(2판)&gt;의 소스코드](https://github.com/joelgrus/data-science-from-scratch)를 
# 기본 틀로 삼아
# 파이썬 언어를 이용한 데이터 분석 기초 강의노트 모음집을 제공합니다.

# **감사의 글**
# 
# 소중한 소스코드를 공개한 웨스 맥키니(Wes McKinney)와 조엘 그루스(Joel Grus)에게 
# 진심어린 감사를 전합니다.
# 
# 
# 머신러닝/딥러닝 기술이 획기적으로 발전하면서 데이터 분석 및 인공지능 관련 연구의 
# 중요성이 사회, 경제, 산업의 거의 모든 분야에 지대한 영향을 미치고 있으며,
# 앞으로 그런 경향이 더욱 강화될 것으로 기대된다.
# 
# 본 강의는 데이터 분석의 기본 아이디어와 다양한 활용법을 실전 예제와 
# 함께 전달한다. 
# 여기서 다루는 내용은 또한 머신러닝/딥러닝 학습을 위한 기초 지식으로 활용된다.

# ## 데이터 과학과 파이썬 데이터 분석

# 데이터 과학이란 **주어진 데이터로부터 수학과 통계 지식을 활용하여 필요한 정보를 추출하는 학문 분야**이다.
# 반면에 파이썬 데이터 분석은 **데이터 과학의 주요 연구 도구**이다. 
# 
# 파이썬 데이터 분석 학습을 위해 아래 분야의 기초지식이 요구되지만
# 미적분학, 선형대수, 확률과통계 관련 이론은 여기서는 필요한 최소의 내용만 다룬다.
# 
# * 파이썬 프로그래밍
# * 미적분학, 선형대수, 확률과통계

# **데이터 과학, 인공지능, 머신러닝, 딥러닝**
# 
# 데이터 과학과 인공지능, 머신러닝, 딥러닝의 관계는 아래 그림으로 설명된다.
# 
# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/datapy/master/jupyter-book/images/ai-ml-relation.png" style="width:500px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="http://www.kyobobook.co.kr/readIT/readITColumnView.laf?thmId=00198&sntnId=14142">교보문고: 에이지 오브 머신러닝</a>&gt;</div></p>

# 위 그림에서 언급된 분야들의 정의는 다음과 같다. 
# 
# * 인공지능: 사고(thinking), 학습(learning) 등 인간의 지적능력을 컴퓨터를 통해 구현하는 
#     기술 또는 해당 연구 분야
# * 머신러닝: 컴퓨터가 데이터로부터 스스로 정보를 추출하는 기법 또는 해당 연구 분야.
# * 딥러닝: 심층 신경망 이론을 기반으로 복잡한 비선형 문제를 해결하는 머신러닝 기법 
#     또는 해당 연구 분야    

# 역사적 관점에서 바라본 인공지능, 머신러닝, 딥러닝의 관계는 다음과 같다.
# 
# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/datapy/master/jupyter-book//images/ai-ml-relation2.png" style="width:600px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://blogs.nvidia.com/blog/2016/07/29/whats-difference-artificial-intelligence-machine-learning-deep-learning-ai/">NVIDIA 블로그</a>&gt;</div></p>

# ## 주요 학습내용

# * 파이썬 기초
# * numpy 활용
# * pandas 활용
# * 데이터 불러오기 및 저장
# * 데이터 전처리: 데이터 정제 및 변환
# * 데이터 다루기: 조인, 병합, 변형
# * 데이터 시각화
# * 데이터 집계와 그룹화
# * 시계열 데이터

# ## 파이썬 프로그래밍 언어

# 프로그래밍 실습에 사용되는 파이썬(Python)은 현재 데이터 분석 및 머신러닝 분야에서 
# 가장 많이 사용되는 프로그래밍언어이다.
# 
# [TIOBE Index](https://www.tiobe.com/tiobe-index/) 2022년 12월 기준 가장 많이 사용되는 프로그래밍 언어이며 
# 점유율이 점점 높아지고 있다. 
# 
# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/datapy/master/jupyter-book/images/tiobe-index.jpg" style="width:750px"></div>

# 파이썬이 데이터 과학 분야에서 인기가 높은 이유는 다음과 같다.
# 
# * 범용 프로그래밍언어
# * R, 매트랩, SQL, 엑셀 등 특정 분야에서 유용하게 사용되는 언어들의 기능 지원
# * 데이터 적재, 시각화, 통계, 자연어 처리, 이미지 처리 등에 필요한 라이브러리 제공
# * 머신러닝, 데이터 분석, 수치 해석, 통계, 확률 등에 활용될 수 있는 라이브러리 및 도구의 지속적 개발
#     * [SciPy](https://scipy.org)
#     * [statsmodels](https://www.statsmodels.org/stable/index.html)
#     * [scikit-learn](https://scikit-learn.org/)
#     * [TensorFlow](https://www.tensorflow.org/)
#     * [Keras](https://keras.io/)
#     * [PyTorch](https://keras.io/)

# ## 프로그래밍 환경

# ### 오프라인 개발환경

# [아나콘다(Anaconda)](https://www.anaconda.com)의 주피터 노트북을 이용한다.
# 콘다 한경(conda environment)을 설정하여 파이썬 코드 실행에 필요한 패키지를 미리 설치하고 관리한다.

# **주피터 노트북 실행환경 준비**

# 아래 명령을 임의의 터미널에서 실행하면 `pydata-book` 콘다 환경을 지정할 수 있다.

# ```bash
# (base) $ conda config --add channels conda-forge
# (base) $ conda config --set channel_priority strict
# (base) $ conda create -y -n pydata-book python=3.10
# (base) $ conda activate pydata-book
# (pydata-book) $ conda install -y pandas jupyter matplotlib
# (pydata-book) $ conda install lxml beautifulsoup4 html5lib openpyxl requests sqlalchemy seaborn scipy statsmodels patsy scikit-learn pyarrow pytables numba
# ```

# 윈도우 사용자의 경우 아나콘다 패키지와 함께 설치된 `Anaconda Prompt (anaconda3)` 터미널을 이용할 수 있다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/datapy/master/jupyter-book/images/anaconda_prompt.jpg" style="width:750px"></div>

# **주피터 노트북 실행**

# `Anaconda Prompt (anaconda3)` 등 임의의 터미널에서 아래 명령을 차례대로 실행하면 주피터 노트북을 열거나 새로 작성할 수 있다.

# ```bash
# (base) $ conda activate pydata-book
# (pydata-book) $ jupyter notebook
# ```

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/datapy/master/jupyter-book/images/anaconda_prompt-1.jpg" style="width:750px"></div>

# 인터넷 브라우저에서 아래와 같은 주피터 서버가 실행되면 원하는 폴더에서 기존의 노트북을 실행, 수정하거나 새로운 주피터 노트북을 생성할 수 있다.
# 
# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/datapy/master/jupyter-book/images/JupyterServer.jpg" style="width:750px"></div>

# ### 온라인 개발환경

# [구글 코랩](https://colab.research.google.com)의 주피터 노트북을 활용하며 특별히 추가 설정을 할 필요가 없지만 
# 특정 라이브러리를 설치해야하는 경우가 발생할 수 있다. 그럴 때는 `pip` 파이썬 라이브러리 관리자를 이용하면 된다.
# 예를 들어 `beautifulsoup4` 라이브러리를 설치하려면 아래 명령문을 주피터 노트북 코드셀에서 실행하면 된다.
# 
# ```bash
# !pip install beautifulsoup4
# ```
# 
# 참고: `pip`은 파이썬 명령어가 아니라 쉘(shell) 명령어이다. 이런 쉘 명령어를 터미널이 아닌 주피터 노트북 셀에서 실행하려면 느낌표(`!`)를 함께 사용한다.
