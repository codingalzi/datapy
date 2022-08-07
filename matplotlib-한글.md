# matplotlib에서 한글 지원하기

`matplotlib` 라이브러리의 API를 이용하여 그래프를 그릴 때 한글 라벨, 타이틀 등을 사용하는 방법을 설명한다.
사용하는 운영체제에 따라 다른 방식을 사용한다. 

**윈도우 운영체제**

먼저 아래 코드를 실행한다. 

```python
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/NGULIM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
```

`NGULIM.TTF`는 새굴림채 보통을 가리키는 폰트이며
임의의 다른 한글 폰트로 지정할 수 있다.
보다 자세한 사항은
[여기](https://bskyvision.com/entry/python-matplotlibpyplot%EB%A1%9C-%EA%B7%B8%EB%9E%98%ED%94%84-%EA%B7%B8%EB%A6%B4-%EB%95%8C-%ED%95%9C%EA%B8%80-%EA%B9%A8%EC%A7%90-%EB%AC%B8%EC%A0%9C-%ED%95%B4%EA%B2%B0-%EB%B0%A9%EB%B2%95)를 참고할 수 있다.

위 단계를 실행했음에도 불구하고 font를 찾을 수 없다는 등의 경고가 발생하면
아래 홈디렉토리의 `.matplotlib` 폴더에 포함된 `fontList*.json` 형식의 파일을 삭제한 후에
주피터노트북을 재실행한다.

`fontList*.json` 파일의 위치를 알아내기 위해 아래 명령문을 이용할 수 있다

```python
import matplotlib as mpl
print(mpl.get_cachedir())
```

**구글코랩 또는 우분투 운영체제**

먼저 아래 코드를 실행한다.
우분투에서는 한 번만 실행하면 되지만
구글코랩에서는 매번 실행해야 한다.

```python
!apt install -y fonts-nanum
!fc-cache -fv
```
주피터노트북 상단에서 아래 명령문을 실행한다.
글씨체(font)는 적절한 폰트로 바꿀 수 있다.

```python
applyfont = "NanumBarunGothic"

import matplotlib.font_manager as fm
if not any(map(lambda ft: ft.name == applyfont, fm.fontManager.ttflist)):
    fm.fontManager.addfont("/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf")

plt.rc("font", family=applyfont)
plt.rc("axes", unicode_minus=False)
```

위 단계를 실행했음에도 불구하고 font를 찾을 수 없다는 등의 경고가 발생하면
윈도우의 경에서 설명한 것처럼 `fontList*.json` 파일을 삭제한다.
우분투의 경우 홈디렉토리의 `.cache/matplotlib` 폴더에 위치할 수 있다.

구글 코랩의 경우 런타임을 다시 시작해야 한다.