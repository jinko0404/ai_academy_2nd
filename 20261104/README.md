# Day 13. OpenCV 이미지·영상 전처리 정리

> Vision·멀티모달 AI 응용 · 2026.11.04 (수)
> 주제: 이미지 표현·색공간 → 필터·형태학 → 영상·증강 → 전처리 프로젝트

## 목차

1. [큰 그림: 왜 전처리인가](#1-큰-그림-왜-전처리인가)
2. [이미지는 숫자 배열이다](#2-이미지는-숫자-배열이다)
3. [이미지 입출력 (imread · imwrite)](#3-이미지-입출력-imread--imwrite)
4. [BGR vs RGB, 채널 분리·병합](#4-bgr-vs-rgb-채널-분리병합)
5. [색공간 변환 (cvtColor)](#5-색공간-변환-cvtcolor)
6. [Resize와 보간법](#6-resize와-보간법)
7. [Filter와 Kernel](#7-filter와-kernel)
8. [Edge 검출 (Canny)](#8-edge-검출-canny)
9. [Threshold (이진화)](#9-threshold-이진화)
10. [Morphology (형태학 처리)](#10-morphology-형태학-처리)
11. [영상 처리 (VideoCapture · VideoWriter)](#11-영상-처리-videocapture--videowriter)
12. [데이터 증강 (Augmentation)](#12-데이터-증강-augmentation)
13. [전처리 프로젝트 구조](#13-전처리-프로젝트-구조)
14. [검증과 코드리뷰 체크리스트](#14-검증과-코드리뷰-체크리스트)
15. [핵심 함수 총정리](#15-핵심-함수-총정리)
16. [복습용 요약](#16-복습용-요약)

---

## 1. 큰 그림: 왜 전처리인가

촬영 조건이 다른 이미지·영상을 **다음 단계(모델·규칙)가 쓸 수 있는 입력으로 준비**하는 것이 전처리다.

```
촬영·수집  →  전처리 (오늘)  →  모델·규칙  →  업무 판단
이미지·영상    표현·크기·검증     분류·검출·인식    검사·분류·확인
```

### 오늘 다루는 처리의 위치

| 단계 | 내용 |
|---|---|
| 입력 | `imread`로 이미지 읽기, 영상은 Frame 반복 |
| 색·크기 | 색공간 변환, Resize |
| 이웃 연산 | Filter, Edge, Threshold, Morphology |
| 영상·증강 | Frame 반복 처리, 학습용 증강 |
| 검증·통합 | 원본 대비 비교, 설정·요약표 저장 |

> 영상도 결국 **"이미지 처리 + 시간 순서 + 저장 설정"**이다. 한 장 처리 함수를 Frame마다 재사용한다.

### 오늘과 내일의 경계

| 오늘 (11/04) | 다음 수업 (11/05) |
|---|---|
| 픽셀·채널·색공간 이해 | 전이학습 기반 Vision 모델, 데이터셋 설계 |
| Resize·필터·이진화·형태학 | Fine-tuning·평가·오류 분석 |
| 영상 Frame 반복 처리·저장 | 클래스 불균형, Grad-CAM |
| 원본 대비 검증 프로젝트 | 오늘 전처리 결과를 모델 입력으로 연결 |

---

## 2. 이미지는 숫자 배열이다

영상에서 한 장을 꺼내면 이미지이고, 이미지는 **위치별 숫자의 배열**이다.

```
영상 ⊃ Frame (H×W 이미지) ⊃ Pixel (y, x 위치의 값) ⊃ 채널 값
```

| 종류 | 형태 |
|---|---|
| Grayscale | `[H, W]` 밝기 배열 |
| Color | `[H, W, C]` 채널 배열 |
| 일반적인 dtype | `uint8` (값 범위 0~255) |
| OpenCV 기본 색상 순서 | **BGR** |

### 먼저 확인할 속성

| 항목 | 의미 | 확인 코드 | 예시 |
|---|---|---|---|
| shape | 행·열·채널의 크기 | `img.shape` | `(345, 450, 3)` |
| dtype | 원소의 자료형 | `img.dtype` | `uint8` |
| min/max | 밝기 값 범위 | `img.min()`, `img.max()` | `0 / 255` |
| ndim | 배열 차원 수 | `img.ndim` | `3` |

- `shape = (H, W, C)` = (세로, 가로, 채널). 예: `(240, 320, 3)`
- 한 픽셀 값은 `img[y, x]`로 확인한다.
- Shape를 모르고 처리하면 채널 축과 공간 축을 뒤바꾸거나 잘못된 크기로 저장할 수 있다.

---

## 3. 이미지 입출력 (imread · imwrite)

```python
from pathlib import Path
import cv2

path = Path("data/sample_landscape.png")
img = cv2.imread(str(path), cv2.IMREAD_COLOR)
if img is None:
    raise RuntimeError("image read failed")

print("shape:", img.shape)
print("dtype:", img.dtype)
print("range:", img.min(), img.max())

Path("outputs").mkdir(parents=True, exist_ok=True)
if not cv2.imwrite("outputs/copy.png", img):
    raise OSError("image save failed")
```

| 함수 | 필수 인자 | 선택 인자 | 반환 |
|---|---|---|---|
| `cv.imread` | `filename` | `flags=IMREAD_COLOR` | `ndarray` 또는 **`None`** (실패 시) |
| `cv.imwrite` | `filename`, `img` | `params=[]` | 저장 성공 여부 `bool` |

| flags | 의미 |
|---|---|
| `IMREAD_GRAYSCALE` | `[H, W]` 1채널로 읽기 |
| `IMREAD_UNCHANGED` | Alpha 채널 유지 가능 |

- 읽기는 파일 내용과 디코더, 저장은 확장자에 맞는 인코더 지원을 확인한다.
- `imread` 직후에는 항상 **`None`, shape, dtype**을 확인한다.

### imshow · waitKey

- `cv.imshow(window_name, img)`: OS 창에 이미지 표시
- `cv.waitKey(delay)`: 키 입력 대기. `delay=0`이면 무기한 대기
- 원격 서버·Notebook·일부 VS Code 환경에서는 **GUI 창이 안 열릴 수 있다.**
- 대안: `cv.imwrite`로 저장하거나, Matplotlib로 RGB 변환 후 표시

---

## 4. BGR vs RGB, 채널 분리·병합

| 도구 | 채널 순서 |
|---|---|
| OpenCV `imread` | **B, G, R** |
| Matplotlib `imshow` | **R, G, B** |

BGR 이미지를 그대로 Matplotlib에 넣으면 **빨강과 파랑이 뒤바뀐 색**으로 보인다.

```python
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # 표시 전에 변환
```

예: 한 픽셀 BGR `[20, 80, 200]` (R 값이 가장 큼) → RGB `[200, 80, 20]`

### split · merge

```python
b, g, r = cv2.split(img)
print(b.shape, g.shape, r.shape)        # 각각 (H, W)

merged = cv2.merge([b, g, r])
print("same:", (merged == img).all())

rgb = cv2.merge([r, g, b])
```

| 함수 | 역할 |
|---|---|
| `cv.split(src)` | 채널별 2차원 배열의 튜플 반환 |
| `cv.merge(mv)` | 동일한 H·W의 채널 배열을 하나의 다채널 이미지로 합침 |

> ⚠️ RGB 배열을 `imwrite`로 저장할 때는 **BGR로 되돌려야** 한다.

---

## 5. 색공간 변환 (cvtColor)

색공간은 "정답"이 아니라 **처리 목적에 따라 고르는 표현 방식**이다.

| 목적 | 표현 선택 |
|---|---|
| Matplotlib에 색 표시 | BGR → RGB |
| 밝기·경계 분석 | Gray |
| 특정 색 범위 선택 | HSV |

| 표현 | 설명 |
|---|---|
| BGR/RGB | 일반적인 컬러 표현 |
| Gray | 밝기 중심의 1채널 표현 |
| HSV (uint8) | H 0~179, S·V 0~255 |
| Binary | 색공간이 아닌 **전경·배경 Mask** |

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # (H, W)
rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    # (H, W, 3)
hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)    # (H, W, 3)
```

| 변환 코드 | 입력 → 출력 | 주요 용도 |
|---|---|---|
| `COLOR_BGR2RGB` | BGR 3채널 → RGB 3채널 | Matplotlib 표시 |
| `COLOR_BGR2GRAY` | BGR 3채널 → Gray 1채널 | Threshold·Edge |
| `COLOR_BGR2HSV` | BGR 3채널 → HSV 3채널 | 색상 범위 분리 |
| `COLOR_GRAY2BGR` | Gray 1채널 → BGR 3채널 | 컬러 도형 그리기 |

`cvtColor` 인자: 필수 `src`, `code` / 선택 `dstCn=0` / 반환 `ndarray`

---

## 6. Resize와 보간법

리사이즈는 크기뿐 아니라 **보간법**도 선택해야 한다.

- **축소**: 여러 픽셀을 하나로 요약
- **확대**: 없는 픽셀 값을 주변에서 추정

| 보간법 | 특징 |
|---|---|
| `INTER_AREA` | **축소**에 자주 사용 |
| `INTER_LINEAR` | 기본값. 주변 4개 값으로 부드럽게 보간. 확대에도 사용 |
| `INTER_NEAREST` | 픽셀 경계가 뚜렷하지만 계단 현상이 커질 수 있음 |

```python
small = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_AREA)

half = cv2.resize(img, dsize=None, fx=0.5, fy=0.5,
                  interpolation=cv2.INTER_AREA)
```

| 인자 | 필수/선택 | 의미 |
|---|---|---|
| `src` | 필수 | 입력 이미지 |
| `dsize` | 필수 자리 | 출력 **(W, H)**. 또는 `None`·`(0,0)` |
| `fx`, `fy` | 선택 | x·y 방향 배율 |
| `interpolation` | 선택 | 픽셀 값 추정 방법 (기본 `INTER_LINEAR`) |

> ⚠️ **`shape`는 `(H, W)`, `dsize`는 `(W, H)`로 순서가 반대다.**
> `dsize`와 `fx·fy`를 동시에 쓰기보다 한 방식만 골라 코드를 명확하게 한다.

---

## 7. Filter와 Kernel

필터는 **작은 이웃 영역의 값을 모아 한 출력 위치의 값을 계산**한다. Kernel이 이웃 영역을 순회한다.

| 종류 | 효과 |
|---|---|
| Blur | 노이즈와 세부를 부드럽게 |
| Sharpen | 경계·고주파 성분 강조 |
| Edge | 밝기 변화가 큰 위치 찾기 |

### 평균 필터 계산 예시

3×3 이웃에서 가운데 값이 100이고 나머지가 10이면:

```
(10×8 + 100) ÷ 9 = 20     # 같은 위치끼리 곱한 뒤 모두 더함
```

### Kernel 크기와 가중치의 영향

| 설정 | 영향 |
|---|---|
| 작은 Kernel | 세부 보존, 약한 효과 |
| 큰 Kernel | 더 강한 Blur, 연산 증가 |
| 합이 1인 평균 Kernel | 밝기 평균화 |
| 음수 가중치 포함 | 경계·선명도 강조 |

> Kernel이 크다고 항상 좋은 것은 아니다. **정보 손실과 연산량이 함께 증가**한다.

### Gaussian · Median · Bilateral 비교

| 필터 | 특징 | 장점 | 주의 |
|---|---|---|---|
| `GaussianBlur` | 가우시안 가중 평균 | 자연스러운 Blur | Edge도 흐려짐 |
| `medianBlur` | 중앙값 선택 | 점 잡음에 강함 | `ksize`는 홀수 |
| `bilateralFilter` | 공간 + 밝기 차이 반영 | **Edge 보존** | 비교적 느림 |

```python
gaussian = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)
median   = cv2.medianBlur(gray, ksize=5)
```

| 함수 | 필수 인자 | 선택 인자 | 핵심 제약 |
|---|---|---|---|
| `GaussianBlur` | `src`, `ksize`, `sigmaX` | `sigmaY=0`, `borderType` | 양수 홀수 또는 0 |
| `medianBlur` | `src`, `ksize` | 없음 | 1보다 큰 홀수 (3, 5, 7...) |
| `bilateralFilter` | `src`, `d`, `sigmaColor`, `sigmaSpace` | `borderType` | `d`가 클수록 연산 증가 |

### Border (가장자리) 처리

Kernel이 이미지 밖으로 나가면 실제 픽셀이 없으므로 규칙이 필요하다. 같은 Kernel이라도 Border 규칙에 따라 경계 결과가 달라진다.

- `BORDER_REFLECT`: 경계 값을 반사하여 확장
- `BORDER_REPLICATE`: 가장자리 픽셀을 반복

---

## 8. Edge 검출 (Canny)

강한 밝기 변화를 찾는다. **Canny 전에 Blur**를 적용하는 것이 일반적이다. 잡음으로 생기는 불필요한 Edge를 줄이기 위해서다.

```python
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, threshold1=70, threshold2=150)
cv2.imwrite("outputs/edges.png", edges)
```

- 필수: `image`, `threshold1`, `threshold2` / 선택: `apertureSize=3`, `L2gradient=False`
- 출력: **0 또는 255의 1채널** Edge 이미지
- Threshold가 **낮으면** 잡음 Edge가 늘고, **높으면** 약한 Edge를 놓칠 수 있다.

---

## 9. Threshold (이진화)

밝기 기준으로 **전경과 배경을 나눈다.** 입력은 보통 Grayscale 1채널이다.

```
THRESH_BINARY, T=128: 값 > T 이면 255, 나머지는 0

Gray   : 40   110   160   220
Binary :  0     0   255   255
```

```python
retval, binary = cv2.threshold(
    gray, thresh=140, maxval=255, type=cv2.THRESH_BINARY
)
print("used threshold:", retval)
```

- 필수 인자: `src`, `thresh`, `maxval`, `type`
- 반환값 **두 개**: `retval`(실제 사용 임계값), `dst`(결과 이미지)
- `type` 예: `THRESH_BINARY`, `THRESH_BINARY_INV`
- Binary 결과는 후속 형태학 처리에 적합하다.

### Fixed · Otsu · Adaptive

| 방법 | 임계값 결정 | 장점 | 주의 |
|---|---|---|---|
| Fixed | 사용자가 한 값 지정 | 단순·빠름 | 조명 변화에 취약 |
| Otsu | 히스토그램으로 자동 결정 | 두 분포가 분리될 때 유용 | Blur가 도움될 수 있음 |
| Adaptive | 지역별 임계값 계산 | 불균일 조명 대응 | `blockSize`·`C` 조정 필요 |

선택 기준: 조명 변화, 전경·배경 분포, 처리 속도, 후속 작업의 요구사항

---

## 10. Morphology (형태학 처리)

Binary 이미지의 **전경 모양을 Kernel로 다듬는** 처리. 실습에서는 **흰색(255)을 전경**으로 둔다.

| 연산 | 순서 | 효과 | 예시 |
|---|---|---|---|
| Erosion (침식) | 침식 1회 | 흰색 전경 **축소** | 붙은 물체 분리 |
| Dilation (팽창) | 팽창 1회 | 흰색 전경 **확대** | 끊긴 부분 연결 |
| **Opening** | Erosion → Dilation | **작은 흰색 잡음 제거** | 먼지 제거 |
| **Closing** | Dilation → Erosion | **작은 검은 구멍 메우기** | 문자 끊김 보완 |

### Kernel 만들기

```python
kernel = cv2.getStructuringElement(
    shape=cv2.MORPH_ELLIPSE, ksize=(5, 5)
)
```

- `shape` 예: `MORPH_RECT`, `MORPH_ELLIPSE`, `MORPH_CROSS`
- Kernel 크기와 **반복 횟수(iterations)**가 실제 영향 범위를 함께 결정한다.

### 적용

```python
opened = cv2.morphologyEx(binary, op=cv2.MORPH_OPEN,
                          kernel=kernel, iterations=1)
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
```

- 필수: `src`, `op`, `kernel` / 선택: `anchor`, `iterations=1`, `borderType`, `borderValue`
- 전경을 흰색으로 두는 관례를 먼저 확인한다.

### 전체 Pipeline 예

```
원본 → Gray → Blur → Threshold → Morphology(Closing 등)
```

처리 순서가 바뀌면 결과도 달라진다. 각 단계의 **입력·출력 shape와 값 범위를 기록**한다.

> ⚠️ **정보 손실 주의**: 작은 흰점이 실제 결함이라면 Opening(3×3 Kernel)이 그 결함을 지울 수 있다. 필요한 작은 특징이 남는지 확인해야 한다.

---

## 11. 영상 처리 (VideoCapture · VideoWriter)

### 영상의 핵심 속성

| 속성 | 의미 | 확인 예 |
|---|---|---|
| Frame | 시간 t의 한 장 이미지 | `cap.read()` |
| FPS | 초당 Frame 수 | `CAP_PROP_FPS` |
| Frame Size | (width, height) | `CAP_PROP_FRAME_WIDTH` |
| Codec | 압축·저장 방식 | `fourcc` |

### Frame 읽기

```python
cap = cv2.VideoCapture("data/input.mp4")
if not cap.isOpened():
    raise RuntimeError("video open failed")

while True:
    ok, frame = cap.read()
    if not ok:          # 마지막 Frame 이후엔 ok=False
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

cap.release()           # 자원 해제
```

| 항목 | 반환·의미 | 주의 |
|---|---|---|
| `VideoCapture(source)` | Capture 객체 | `0`은 기본 카메라 |
| `isOpened()` | 열렸는지 `bool` | |
| `read()` | `(ok, frame)` | 마지막 Frame 후 `ok=False` |
| `get(property id)` | 속성 값 (float) | 코덱에 따라 값이 부정확할 수 있음 |
| `set(id, value)` | 설정 성공 여부 | 모든 백엔드가 지원하지는 않음 |
| `release()` | 파일·카메라 자원 해제 | |

### Frame 샘플링

- **Frame index 기반**: `frame_idx % step == 0`일 때만 처리
- **시간 기반**: `CAP_PROP_POS_MSEC` 또는 FPS로 Timestamp 계산
- Frame을 건너뛰면 처리량은 줄지만 **빠른 사건을 놓칠 수 있다.**
- Frame을 줄이거나 저장 FPS를 바꾸면 **출력 재생시간**도 달라질 수 있다.

### VideoWriter로 저장

```python
fps = cap.get(cv2.CAP_PROP_FPS)
if not (0 < fps < float("inf")):
    raise ValueError("invalid FPS")

h, w = frame.shape[:2]
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter("outputs/processed.mp4", fourcc, fps, (w, h), True)
if not writer.isOpened():
    raise RuntimeError("writer open failed")

# Frame 반복문 안에서
writer.write(frame)

# 반복이 끝난 뒤
writer.release()
```

| 인자 | 필수/선택 | 역할 |
|---|---|---|
| `filename` | 필수 | 출력 파일 경로 (확장자 포함) |
| `fourcc` | 필수 | 코덱 식별자 (`mp4v`, `XVID` 등) |
| `fps` | 필수 | 재생 속도 |
| `frameSize` | 필수 | 모든 Frame의 **(W, H)** |
| `isColor` | 선택 | 컬러 Frame 여부 |

### 재생시간 계산 (일정한 FPS 가정)

| 계산 | 결과 |
|---|---|
| 30장 ÷ 30FPS | 1초 |
| 15장 ÷ 15FPS | 1초 |
| 15장 ÷ 30FPS | 0.5초 |

같은 Frame 수에서 출력 FPS를 절반으로 하면 재생 시간이 길어져 느리게 보인다. **저장 FPS와 연산 처리속도는 다른 개념**이다.

### 영상 저장 오류를 찾는 순서

1. `writer.isOpened()`가 `True`인가?
2. Frame의 `(width, height)`가 `frameSize`와 일치하는가?
3. Gray Frame을 쓰면서 `isColor=True`로 두지 않았는가?
4. 출력 확장자·fourcc·설치 코덱의 조합이 맞는가?
5. `release()` 후 파일 크기와 다시 읽기 여부를 확인했는가?

---

## 12. 데이터 증강 (Augmentation)

**라벨 의미를 유지하는 변형**을 추가해 학습 자료의 다양성을 늘린다. (Flip, Crop, Rotation, Brightness, Blur 등)

### 가장 중요한 원칙

```
원본 기준으로 먼저 분할
 ├─ train       → 기본 전처리 + 무작위 증강 → 모델 학습
 └─ valid/test  → 정해진(결정적) 전처리만 → 같은 평가 조건
```

- valid/test에 무작위 증강을 매번 적용하면 **평가 입력이 매번 달라져 성능 비교가 불안정**해진다.
- 라벨 의미가 바뀌는 변형은 제외한다.

### 주요 변형 함수

```python
# 반전
horizontal = cv2.flip(img, flipCode=1)   # 1=좌우, 0=상하, 음수=모두

# 회전
h, w = img.shape[:2]
M = cv2.getRotationMatrix2D(center=(w / 2, h / 2), angle=10, scale=1.0)
rotated = cv2.warpAffine(img, M, dsize=(w, h), borderMode=cv2.BORDER_REFLECT)

# 밝기·Crop·Blur
bright = cv2.convertScaleAbs(img, alpha=1.0, beta=30)
crop = img[h//10:h-h//10, w//10:w-w//10]
crop = cv2.resize(crop, (w, h))           # 자른 뒤 원래 크기로
blur = cv2.GaussianBlur(img, (5, 5), 0)
```

| 함수 | 설명 |
|---|---|
| `cv.flip(src, flipCode)` | 좌우·상하 반전 |
| `getRotationMatrix2D(center, angle, scale)` | 회전 **2×3 행렬** 생성 (angle은 도 단위) |
| `warpAffine(src, M, dsize)` | 행렬로 실제 픽셀 재배치. 회전 후 모서리 손실·Border 확인 |
| `convertScaleAbs(alpha, beta)` | 밝기 조정, 절댓값·uint8 변환도 수행 |

### 과제별 증강 가이드

| 과제 | 사용 가능성이 높은 변형 | 주의할 변형 |
|---|---|---|
| 일반 사물 분류 | 좌우 반전, 밝기, 작은 Crop | 큰 회전, 과도한 Blur |
| 문자 OCR | 작은 회전, 명암, 노이즈 | **좌우·상하 반전** (문자 모양·의미 변경) |
| 교통 표지 | 밝기, 날씨 효과 | 좌우 반전이 의미를 바꿀 수 있음 |
| 의료 영상 | 도메인 기준에 따른 제한적 변형 | 임의 색상·방향 변경 |

### 재현 가능한 증강 Pipeline

- 변환 설정을 dict 또는 JSON으로 기록한다.
- **Seed·난수 도구·호출 순서·버전**을 함께 기록한다. (같은 실험을 재현하고 설정 차이를 분리하기 위해)
- 원본 단위로 분할한 뒤 **train에만** 무작위 변환을 적용한다.
- 원본 파일은 보존하고, 결과 파일명에 변환 정보를 남긴다.

---

## 13. 전처리 프로젝트 구조

### 환경과 원칙

```bash
python --version                                  # Python 3.13.x
python -m pip show opencv-python numpy matplotlib
python -c "import cv2; print(cv2.__version__)"    # OpenCV 4.x import 성공
```

- 원본 `data/`는 **수정하지 않는다.**
- 생성 결과는 `outputs/`에 저장한다.
- 서버·원격 환경에서는 `imshow` 대신 `imwrite` 또는 Matplotlib 저장을 쓴다.

### 권장 폴더 구조

```
vision_preprocess/
├── README.md
├── main.py            # 설정 읽기·작업 선택·전체 실행
├── image_ops.py       # 색공간·Resize·Filter·Morphology
├── video_ops.py       # Frame 반복·Writer 저장
├── augment.py         # 학습용 변형 함수
├── data/
│   ├── images/
│   └── videos/
├── outputs/
│   ├── images/
│   └── videos/
└── reports/
    └── result_summary.csv   # 원본·출력 Shape·설정·성공 여부
```

### 함수의 입력·출력·부작용을 명확히

| 함수 | 입력 | 출력 | 파일 I/O |
|---|---|---|---|
| `load_image` | `Path` | BGR ndarray | 파일 읽기 |
| `preprocess_image` | ndarray, config | processed ndarray | 없음 |
| `save_image` | `Path`, ndarray | `bool` 또는 `Path` | 파일 생성 |
| `process_video` | input, output, config | summary dict | 영상 생성 |
| `augment_image` | ndarray, seed | augmented ndarray | 없음 |

### 폴더 일괄 처리 예

```python
from pathlib import Path
import cv2

src_dir = Path("data/images")
out_dir = Path("outputs/images")
out_dir.mkdir(parents=True, exist_ok=True)

for path in sorted(src_dir.glob("*.png")):     # sorted로 순서 고정 → 재현성
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"read failed: {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (320, 240))
    if not cv2.imwrite(str(out_dir / path.name), resized):
        raise OSError(f"save failed: {path}")
```

### 영상 Frame 처리 함수

```python
def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, 70, 150)
    return cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)   # Writer와 같은 shape·채널

while True:
    ok, frame = cap.read()
    if not ok:
        break
    result = process_frame(frame)
    writer.write(result)
```

> **원칙**: Frame 처리 함수는 파일을 열고 닫지 않는다. 반복과 I/O는 상위 함수가 담당한다.
> (Frame 처리 함수 안에서 `VideoCapture`를 매번 열면 반복 I/O와 자원 관리가 복잡해지고 성능이 크게 떨어진다.)

---

## 14. 검증과 코드리뷰 체크리스트

### 원본 대비 검증 항목

파일이 열리더라도 작업에 필요한 **경계·글자·관심 영역이 남아야** 한다.

| 검증 항목 | 원본 예시 | 결과 예시 | 판정 |
|---|---|---|---|
| 파일 수 | 10 | 10 | 일치 |
| Shape | 다양함 | 240×320 | 설정 일치 |
| dtype | uint8 | uint8 | 유지 |
| 값 범위 | 0~255 이내 | 0~255 이내 | 범위 확인 |
| 읽기 재확인 | 가능 | 가능 | 정상 |

영상은 **Frame 수·FPS·재생시간**, 저장은 **파일 수·재읽기**를 확인한다.

### 정상·경계·실패 테스트 케이스

| 상황 | 기대 결과 | 검증 |
|---|---|---|
| 정상 PNG | 읽기·처리·저장 성공 | shape·dtype·재읽기 |
| 다른 크기 이미지 | 설정 크기로 통일 | 원본 비율·왜곡 확인 |
| Gray 입력 | 채널 처리 분기 | BGR 변환 필요 여부 |
| 빈 영상/손상 파일 | 명확한 실패 기록 | 출력 생성 금지 |
| FPS 0 또는 미확인 | 대체 규칙 또는 중단 | 보고서 기록 |
| 출력 폴더 없음 | 폴더 생성 후 저장 | 파일 수 확인 |

### 코드리뷰 체크리스트

- [ ] 입력 파일을 읽은 직후 `None`·shape·dtype을 확인했는가?
- [ ] BGR/RGB와 (H,W)/(W,H)의 순서를 혼동하지 않았는가?
- [ ] 필터·Threshold·Morphology 설정을 코드나 JSON에 남겼는가?
- [ ] 원본 파일을 보존하고 출력 폴더를 분리했는가?
- [ ] 영상 Writer의 FPS·Frame Size·채널이 일치하는가?
- [ ] README만으로 다른 사람이 같은 결과를 만들 수 있는가?

### 보고서에 최소한 기록할 것

입력 경로, 처리 설정, 출력 Shape/성공 여부 (원본을 덮어쓰지 않는 이유: 비교·재현·복구가 어려워지기 때문)

### 프로젝트 산출물

| 파일·폴더 | 내용 |
|---|---|
| `main.py` | 전체 실행 진입점 (기본필수는 이 한 파일로 가능) |
| `image_ops.py` / `video_ops.py` | 모듈 분리 시 이미지·영상 처리 |
| `outputs/` | 처리 이미지·영상 |
| `result_summary.csv` | 입출력 수치·설정·성공 여부 |
| `README.md` | 실행법·처리 순서·결과 해석·한계 |

---

## 15. 핵심 함수 총정리

설명할 수 있어야 하는 함수와 인자:

| 영역 | 핵심 함수 | 설명할 수 있어야 할 인자 |
|---|---|---|
| 입출력 | `imread`, `imwrite` | `filename`, `flags`, `img` |
| 색공간 | `cvtColor` | `src`, `code` |
| 크기 | `resize` | `dsize`, `fx`, `fy`, `interpolation` |
| 필터 | `GaussianBlur`, `medianBlur` | `ksize`, `sigmaX` |
| 이진화 | `threshold` | `thresh`, `maxval`, `type` |
| 형태학 | `morphologyEx` | `op`, `kernel`, `iterations` |
| 영상 | `VideoCapture`, `VideoWriter` | `source`, `fps`, `frameSize`, `fourcc` |
| 증강 | `flip`, `warpAffine` | `flipCode`, `M`, `dsize` |

---

## 16. 복습용 요약

### 헷갈리기 쉬운 순서 모음

| 대상 | 순서 |
|---|---|
| `img.shape` | `(H, W, C)` |
| `cv.resize`의 `dsize` | `(W, H)` |
| `VideoWriter`의 `frameSize` | `(W, H)` |
| OpenCV 컬러 | B, G, R |
| Matplotlib 컬러 | R, G, B |
| PyTorch `Conv2d` 배치 입력 | `[N, C, H, W]` |

### 퀴즈 정답 모음

| 질문 | 정답 |
|---|---|
| shape `(345,450,3)`의 450은? | Width (열의 수) |
| `imread` 실패 시 반환값은? | `None` |
| Gray 변환 후 채널 수는? | 1채널 (배열 shape는 보통 `(H, W)`) |
| 축소에 권장되는 보간법은? | `INTER_AREA` |
| `medianBlur`의 `ksize`는? | 1보다 큰 홀수 |
| Opening의 연산 순서는? | Erosion → Dilation |
| Closing이 보완하는 문제는? | 전경 내부의 작은 구멍·끊김 |
| `threshold`의 반환값 두 개는? | `retval`(실제 임계값), `dst`(결과 이미지) |
| `cap.read()`의 반환값은? | `ok`, `frame` |
| 출력 FPS를 절반으로 하면? | 같은 Frame 수라면 재생 시간이 길어져 느리게 보임 |
| valid/test에 무작위 증강을 쓰면? | 평가 입력이 매번 달라져 성능 비교가 불안정 |
| OCR에서 좌우 반전을 조심하는 이유는? | 문자 모양과 의미가 바뀔 수 있어서 |

### 오늘의 핵심 5가지

- **표현**: 이미지는 NumPy 배열이며 OpenCV 컬러 기본값은 BGR이다.
- **변환**: 색공간·Resize·Filter·Threshold·Morphology는 목적과 인자를 기록한다.
- **영상**: Frame 반복 처리와 FPS·크기·코덱 설정을 함께 관리한다.
- **증강**: 라벨 의미를 보존하고 train에서만 확률적 변환을 사용한다.
- **재현**: 원본·결과·설정·검증표·README를 함께 남긴다.

### 다음 수업 연결

| 일자 | 학습 내용 | 오늘과의 연결 |
|---|---|---|
| 11/05 (목) | 전이학습 모델·데이터셋 설계 | 전처리·Resize·Normalize 기준 |
| 11/06 (금) | YOLO 객체탐지·IoU·mAP | Frame·좌표·영상 처리 |
| 11/09 (월) | OCR·VLM·추론 최적화 | Gray·Threshold·형태학·영상 Pipeline |
| 11/10~ | FastAPI·LLM·RAG | Vision 결과를 API와 서비스로 확장 |

### 공식 참고자료

설치한 버전의 **OpenCV 4.x 공식 문서**를 최종 기준으로 확인한다.

- imgcodecs (`imread` / `imwrite`)
- Color Space Conversions (`cvtColor`)
- Geometric Transformations (`resize` / `warpAffine`)
- Smoothing Images (Gaussian / Median / Bilateral)
- Thresholding (`threshold` / `adaptiveThreshold`)
- Morphological Transformations (`erode` / `dilate` / `morphologyEx`)
- Video I/O (`VideoCapture` / `VideoWriter`)
- Affine Transformations (`getRotationMatrix2D` / `warpAffine`)
- Core Array Operations (`split` / `merge` / `min` / `max`)
