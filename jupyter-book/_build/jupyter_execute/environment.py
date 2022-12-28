#!/usr/bin/env python
# coding: utf-8

# # 프로그래밍 개발 환경

# ## 오프라인 개발 환경: 아나콘다

# ### 파이썬 설치

# 데이터 분석 관련 중요 라이브러리, 스파이더 편집기, 주피터 노트북을 모두 포함하는
# [아나콘다<font size='2'>Anaconda</font> 파이썬 패키지](https://www.anaconda.com)를 이용한다.
# 자세한 설치 요령은 [아나콘다 설치](https://codingalzi.github.io/pybook/environment_setting.html#sec-anaconda-install)를
# 참고한다.

# ### 콘다 환경 준비

# 콘다 환경<font size='2'>conda environment</font>으로 가상 환경<font size='2'>virtual environment</font>을 설정하여 
# 파이썬 코드 실행에 필요한 모든 패키지를 설치하고 관리한다.

# :::{admonition} 콘다 환경
# :class: info
# 
# 가상 환경 개념과 아나콘다를 활용한 파이썬 가상환경을 설정하는 방법에 대한 간략한 소개는
# [Anaconda를 활용한 python 가상환경(virtual env) 설정하기](https://teddylee777.github.io/python/anaconda-%EA%B0%80%EC%83%81%ED%99%98%EA%B2%BD%EC%84%A4%EC%A0%95-%ED%8C%81-%EA%B0%95%EC%A2%8C)를
# 참고한다.
# 콘다 환경 관리에 대한 보다 자세한 설명은 [콘다 공식 문서](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)를 참고한다.
# :::

# 임의의 터미널<font size='2'>terminal</font>에서 
# 아래 쉘<font size='2'>shell</font> 명령을 차례대로 실행하면 
# `pydata-book` 콘다 환경을 설정한다.

# ```bash
# (base) $ conda config --add channels conda-forge
# (base) $ conda config --set channel_priority strict
# (base) $ conda create -y -n pydata-book python=3.10
# (base) $ conda activate pydata-book
# (pydata-book) $ conda install -y pandas jupyter matplotlib
# (pydata-book) $ conda install lxml beautifulsoup4 html5lib openpyxl requests sqlalchemy seaborn scipy statsmodels patsy scikit-learn pyarrow pytables numba
# ```

# :::{admonition} 쉘과 터미널
# :class: info
# 
# 쉘<font size='2'>shell</font>은 컴퓨터 운영체제의 기능과 서비스를 관리하거나 
# 구현하는 데에 사용되는 명령문<font size='2'>command lines</font>을 실행하는 
# 실행기<font size='2'>interpreter</font>이다.
# 즉, 사용자와 컴퓨터 커널<font size='2'>kernel</font>을 연결하는 다리 역할을 수행한다.
# 쉘이 실행하는 명령문을 쉘 스크립트<font size='2'>shell script</font>라 부른다.
# 다양한 종류의 쉘이 존재하지만, 
# 리눅스의 기본 쉘인 bash(배시 쉘)와 맥 OSX의 zsh(Z쉘)이 가장 많이 사용된다.
# 
# 반면에 터미널<font size='2'>terminal</font>은 쉘 스크립트를 작성하고 실행시킬 수 있는 
# 사용자 인터페이스<font size='2'>user interface</font>(UI)이며
# 콘솔<font size='2'>console</font>이라고도 불린다.
# :::

# 윈도우 사용자의 경우 아나콘다 패키지와 함께 설치된 `Anaconda Prompt (anaconda3)` 터미널을 이용할 수 있다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/datapy/master/jupyter-book/images/anaconda_prompt.jpg" style="width:750px"></div>
# 

# ### 주피터 노트북 실행

# `Anaconda Prompt (anaconda3)` 등 임의의 터미널에서 아래 명령을 차례대로 실행하면 주피터 노트북을 열거나 새로 작성할 수 있다.

# ```bash
# (base) $ conda activate pydata-book
# (pydata-book) $ jupyter notebook
# ```

# 아래 사진은 Anaconda Prompt (anaconda3) 터미널에서 위 두 명령문을 실행하는 것을 보여준다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/datapy/master/jupyter-book/images/anaconda_prompt-1.jpg" style="width:750px"></div>

# 인터넷 브라우저에서 아래와 같이 주피터 서버가 실행되면 원하는 폴더에서 기존의 노트북을 실행, 수정하거나 새로운 주피터 노트북을 생성할 수 있다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/datapy/master/jupyter-book/images/JupyterServer.jpg" style="width:750px"></div>

# :::{admonition} 주피터 노트북 사용법
# :class: info
# 
# [주피터 노트북 실행법](https://codingalzi.github.io/pybook/environment_setting.html#sec-jupyter-notebook)에서
# 간단한 사용법을 확인할 수 있다. 
# 아나콘다의 설치 과정과 주피터 노트북의 기초 사용법에 대한 보다 상세한 설명은 
# [유튜버 나도코딩의 동영상](https://www.youtube.com/watch?v=dJfq-eCi7KI&t=2298s)을 참고하면 좋다.
# :::

# ## 온라인 개발 환경: 구글 코랩

# [구글 코랩<font size="2">Google Colab</font>](https://colab.research.google.com/?hl=ko)은 
# 구글에서 제공하는 파이썬 전용 온라인 주피터 노트북이다.
# 웹브라우저를 이용하여 어떤 준비 없이 바로 파이썬 프로그래밍을 시작할 수 있다.

# [구글 코랩 사용법](https://codingalzi.github.io/pybook/environment_setting.html#sec-google-colab)에서
# 간단한 사용법을 확인할 수 있다.
# 보다 자세한 설명은 [유튜버 봉수골 개발자 이선비의 동영상](https://www.youtube.com/watch?v=91E0qenm7W4)을 참고한다.
# 그리고 입문용은 아니지만 구글 코랩만이 지원하는 유용한 고급 기능을 
# [TensorFlow 팀의 동영상](https://www.youtube.com/watch?v=rNgswRZ2C1Y)에서 확인할 수 있다.

# 구글 코랩을 사용하다 보면 
# 특정 라이브러리를 설치해야하는 경우가 발생할 수 있는데 그럴 때는 `pip` 파이썬 라이브러리 관리자를 
# 이용하여 설치하면 된다.
# 예를 들어 `beautifulsoup4` 라이브러리를 설치하려면 아래 명령문을 주피터 노트북 코드셀에서 실행한다.
# 
# ```bash
# !pip install beautifulsoup4
# ```

# :::{admonition} pip 라이브러리 관리자
# :class: info
# 
# `pip`은 파이썬 명령어가 아니라 컴퓨터 운영체제 관리에 사용되는 쉘<font size='2'>shell</font> 명령어이다. 
# 이런 쉘 명령어를 터미널이 아닌 주피터 노트북 셀에서 실행하려면 느낌표(`!`)를 함께 사용한다.
# :::

# ## IPython 기초

# 참고: https://compmath.korea.ac.kr/appmath/index.html
