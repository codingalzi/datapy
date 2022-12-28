# 소개

[&lt;파이썬 라이브러리를 활용한 데이터 분석(2판)&gt;의 소스코드](https://github.com/wesm/pydata-book)를 
담고 있는 주피터 노트북과
[&lt;밑바닥부터 시작하는 데이터 과학(2판)&gt;의 소스코드](https://github.com/joelgrus/data-science-from-scratch)를 
기본 틀로 삼아
파이썬 언어를 이용한 데이터 분석 기초 강의노트 모음집을 제공합니다.

**감사의 글**

소중한 소스코드를 공개한 웨스 맥키니(Wes McKinney)와 조엘 그루스(Joel Grus)에게 
진심어린 감사를 전합니다.


머신러닝/딥러닝 기술이 획기적으로 발전하면서 데이터 분석 및 인공지능 관련 연구의 
중요성이 사회, 경제, 산업의 거의 모든 분야에 지대한 영향을 미치고 있으며,
앞으로 그런 경향이 더욱 강화될 것으로 기대된다.

본 강의는 데이터 분석의 기본 아이디어와 다양한 활용법을 실전 예제와 
함께 전달한다. 
여기서 다루는 내용은 또한 머신러닝/딥러닝 학습을 위한 기초 지식으로 활용된다.


**데이터 과학과 파이썬 데이터 분석**

데이터 과학이란 **주어진 데이터로부터 수학과 통계 지식을 활용하여 필요한 정보를 추출하는 학문 분야**이다.
반면에 파이썬 데이터 분석은 **데이터 과학의 주요 연구 도구**이다. 

파이썬 데이터 분석 학습을 위해 아래 분야의 기초지식이 요구되지만
미적분학, 선형대수, 확률과통계 관련 이론은 여기서는 필요한 최소의 내용만 다룬다.

* 파이썬 프로그래밍
* 미적분학, 선형대수, 확률과통계


**데이터 과학, 인공지능, 머신러닝, 딥러닝**

데이터 과학과 인공지능, 머신러닝, 딥러닝의 관계는 아래 그림으로 설명된다.

<img src="https://raw.githubusercontent.com/codingalzi/datapy/master/jupyter-book/images/ai-ml-relation.png" style="width:500px;">

그림 출처: [교보문고: 에이지 오브 머신러닝](http://www.kyobobook.co.kr/readIT/readITColumnView.laf?thmId=00198&sntnId=14142)

위 그림에서 언급된 분야들의 정의는 다음과 같다. 

* 인공지능: 사고(thinking), 학습(learning) 등 인간의 지적능력을 컴퓨터를 통해 구현하는 
    기술 또는 해당 연구 분야
* 머신러닝: 컴퓨터가 데이터로부터 스스로 정보를 추출하는 기법 또는 해당 연구 분야.
* 딥러닝: 심층 신경망 이론을 기반으로 복잡한 비선형 문제를 해결하는 머신러닝 기법 
    또는 해당 연구 분야    

역사적 관점에서 바라본 인공지능, 머신러닝, 딥러닝의 관계는 다음과 같다.

<img src="https://raw.githubusercontent.com/codingalzi/datapy/master/jupyter-book//images/ai-ml-relation2.png" style="width:600px;">

그림 출처: [NVIDIA 블로그](https://blogs.nvidia.com/blog/2016/07/29/whats-difference-artificial-intelligence-machine-learning-deep-learning-ai/)


**주요 학습내용**

* numpy 기초
* pandas 기초
* 데이터 불러오기 및 저장
* 데이터 전처리: 데이터 정제 및 변환
* 데이터 다루기: 조인, 병합, 변형
* 데이터 시각화
* 데이터 집계와 그룹화
* 시계열 데이터
* 고급 pandas
* 모델링 라이브러리: statsmodels, scikit-learn
* 데이터 분석 실전 예제


**파이썬<font size='2'>Python</font> 프로그래밍 언어**

프로그래밍 실습에 사용되는 파이썬(Python)은 현재 데이터 분석 및 머신러닝 분야에서 
가장 많이 사용되는 프로그래밍언어이다.

TIOBE Index 2022년 2월 기준 가장 많이 사용되는 프로그래밍 언어이며 
점유율이 점점 높아지고 있다. 

<img src="https://raw.githubusercontent.com/codingalzi/datapy/master/jupyter-book/images/tiobe-index.jpg" style="width:750px">

그림출처: [TIOBE Index](https://www.tiobe.com/tiobe-index/)

파이썬이 데이터 과학 분야에서 인기가 높은 이유는 다음과 같다.

* 범용 프로그래밍언어
* R, 매트랩, SQL, 엑셀 등 특정 분야에서 유용하게 사용되는 언어들의 기능 지원
* 데이터 적재, 시각화, 통계, 자연어 처리, 이미지 처리 등에 필요한 라이브러리 제공
* 머신러닝, 데이터 분석 등에 활용될 수 있는 라이브러리 및 도구의 지속적 개발
    * [scikit-learn](https://scikit-learn.org/)
    * [TensorFlow](https://www.tensorflow.org/)
    * [Keras](https://keras.io/)
    * [PyTorch](https://keras.io/)
    * ...
