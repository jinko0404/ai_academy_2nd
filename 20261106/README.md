# Day 15 — 객체 탐지(Object Detection)·YOLO·객체 추적(Tracking) 개념 정리

> **학습 자료:** `Day15_20261106.pdf` (2026-11-06, Vision·멀티모달 AI 응용, 106쪽)  
> **목적:** 실습 코드보다 **기본 개념, 핵심 용어, 지표의 의미와 차이**를 빠르게 복습하기 위한 README.  
> **범위:** 1~8교시의 객체 탐지, 좌표, 평가, 라벨링, YOLO 구조·학습·추론, 추적, 성능 측정.

## 1. 전체 흐름 먼저 이해하기

```text
이미지·영상 입력
    ↓
전처리(크기 조정·정규화)
    ↓
YOLO 모델: Backbone → Neck → Head
    ↓
객체 후보: Bounding Box + Class + Confidence
    ↓
점수 필터링 + NMS(중복 제거)
    ↓
탐지 결과(Detect)
    ↓ 영상일 경우
Tracker: 프레임 간 같은 객체 연결 → Track ID
    ↓
CSV·이미지·영상 저장 / 오류 분석 / 성능 평가
```

- **분류(Classification):** 이미지 전체가 무엇인지 예측한다.
- **객체 탐지(Detection):** 이미지 안의 각 객체가 **무엇이고 어디에 있는지** 예측한다.
- **분할(Segmentation):** 객체 또는 영역을 **픽셀 단위 Mask**로 표현한다.
- **객체 추적(Tracking):** 영상의 연속된 프레임에서 **같은 객체를 연결**한다.

## 2. 탐지 결과를 구성하는 기본 용어

| 용어 | 뜻 | 기억할 점 |
|---|---|---|
| Object | 탐지 대상 객체 | 사람, 차량, 안전모 등 |
| Bounding Box (BBox) | 객체를 감싸는 사각형 | 객체 위치와 크기를 표현 |
| Class / Class ID | 객체 종류 / 숫자 식별자 | `0: helmet`, `2: person` 등 |
| Confidence (`conf`) | 탐지 후보의 예측 점수 | 후보 선별·정렬에 사용 |
| Frame | 영상의 한 장면 | 영상에서는 프레임마다 탐지 |
| Image ID / Frame index | 어느 입력에서 나온 결과인지 | 저장·분석 시 식별용 |
| Track ID | 프레임 간 동일 객체를 연결하는 번호 | Class ID와 다름; 추적 중 부여 |
| Ground Truth (GT) | 정답 라벨 | 모델 예측과 비교하는 기준 |
| Prediction | 모델 예측 | 박스·클래스·점수 등 |

**중요:** `Class ID`는 **종류**, `Track ID`는 **영상 속 개별 객체의 연속성**을 나타낸다. 탐지가 없는 프레임도 정상적인 결과이며 객체 수는 0일 수 있다.

## 3. Bounding Box와 좌표계

| 좌표 표현 | 구성 | 주 사용처 |
|---|---|---|
| `xyxy` | 좌상단 `(x1,y1)`, 우하단 `(x2,y2)` | 박스 그리기, IoU |
| OpenCV Rect | 좌상단 `(x,y)` + `width,height` | OpenCV 사각형 |
| 중심 `xywh` | 중심 `(cx,cy)` + `width,height` | `boxes.xywh` |
| Normalized | 이미지 너비·높이로 나눈 0~1 좌표 | YOLO 라벨 |

**픽셀 좌표 → YOLO 정규화 좌표** (`W`: 이미지 너비, `H`: 이미지 높이)

```text
cx = (x1 + x2) / (2 × W)
cy = (y1 + y2) / (2 × H)
w  = (x2 - x1) / W
h  = (y2 - y1) / H
```

예시: 이미지 `640×480`, 박스 `(64,48)~(320,288)` → `(cx,cy,w,h)=(0.30,0.35,0.40,0.50)`.

- 좌표는 **원본 이미지의 W·H**를 기준으로 정규화한다.
- W와 H를 바꾸면 박스의 위치·크기가 잘못된다.
- OpenCV의 색상 채널 순서는 **BGR**이다(RGB와 구별).

## 4. IoU·NMS·Confidence — 반드시 구분

### IoU (Intersection over Union)
두 박스가 얼마나 겹치는지를 나타내는 비율.

```text
IoU = 교집합 면적 / 합집합 면적
합집합 면적 = 박스 A 면적 + 박스 B 면적 - 교집합 면적
```

- **0:** 겹치지 않음 / **1:** 완전히 일치.
- **평가 IoU:** 예측 박스와 **정답 박스**를 비교하여 TP 매칭 판단.
- **NMS IoU:** **예측 박스끼리** 비교하여 중복 후보 제거.
- IoU가 높아도 **클래스 일치 여부는 별도로** 확인한다.

### NMS (Non-Maximum Suppression, 비최대 억제)
동일 객체에 겹쳐 나온 **중복 탐지 후보**를 정리하는 후처리.

1. Confidence가 기준보다 낮은 후보를 제거한다.
2. 남은 후보를 Confidence 내림차순으로 정렬한다.
3. 가장 점수가 높은 후보를 선택한다.
4. 같은 클래스에서 선택된 박스와 IoU가 큰 중복 후보를 제거한다.
5. 후보가 없어질 때까지 반복한다.

| 설정 | 낮게 하면 | 높게 하면 |
|---|---|---|
| Confidence threshold | 후보 증가, Recall 증가 가능, FP 증가 가능 | 후보 감소, Precision 증가 가능, FN 증가 가능 |
| NMS IoU threshold | 조금 겹쳐도 제거, 밀집 객체 누락 위험 | 많이 겹쳐도 유지, 중복 박스 잔존 위험 |

> **암기:** Confidence = **후보 점수**, NMS IoU = **예측끼리 중복 제거**, 평가 IoU = **예측과 정답의 위치 일치**.

## 5. 평가 지표: TP·FP·FN·Precision·Recall·AP·mAP

| 용어 | 의미 | 예시 |
|---|---|---|
| TP (True Positive) | 클래스와 평가 IoU 조건을 만족하여 정답과 1:1 매칭된 예측 | 사람을 올바르게 탐지 |
| FP (False Positive) | 정답과 매칭되지 않은 예측 | 배경을 사람으로 탐지, 잘못된 클래스, 중복 박스 |
| FN (False Negative) | 존재하지만 매칭되는 예측이 없는 정답 | 실제 사람을 놓침 |
| Precision (정밀도) | 탐지했다고 한 것 중 맞은 비율 | `TP/(TP+FP)` |
| Recall (재현율) | 실제 객체 중 찾아낸 비율 | `TP/(TP+FN)` |
| PR Curve | 점수 기준을 바꾸며 그린 Precision–Recall 관계 | AP 계산에 사용 |
| AP (Average Precision) | **한 클래스**의 PR 곡선을 요약한 값 | 클래스별 탐지 성능 |
| mAP (mean AP) | 클래스별 AP 평균 | 여러 클래스의 전체 성능 |
| mAP50 | 평가 IoU **0.50**에서 계산한 mAP | 상대적으로 관대한 위치 기준 |
| mAP50-95 | IoU **0.50~0.95**, 0.05 간격의 10개 기준 평균 | 위치 정확도까지 더 엄격히 평가 |

**계산 예:** TP=8, FP=2, FN=4라면 Precision=`8/10=0.80`, Recall=`8/12≈0.67`.

- **Precision 낮음:** 거짓 탐지(FP) 점검.
- **Recall 낮음:** 놓친 객체(FN) 점검.
- **mAP50 높고 mAP50-95 낮음:** 박스 위치 정밀도 부족 가능성.
- 한 정답에 중복 박스가 여러 개 매칭 후보로 나타나도 **일반적으로 하나만 TP**, 나머지는 FP.

## 6. 라벨링과 YOLO 데이터 형식

### YOLO 라벨 TXT

```text
class_id cx cy w h
0 0.300000 0.350000 0.400000 0.500000
2 0.720000 0.600000 0.180000 0.240000
```

- **객체 하나당 한 줄**, 숫자 **5개**.
- `class_id`는 **0부터 시작하는 정수**.
- 나머지 좌표는 원본 이미지 크기로 **0~1 정규화**.
- 이미지 `scene.jpg` ↔ 라벨 `scene.txt`처럼 **같은 파일명** 사용.
- TXT에는 헤더·주석을 넣지 않는다.

### 데이터셋 구조

```text
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/       # 선택
├── labels/
│   ├── train/
│   ├── val/
│   └── test/       # test를 사용하는 경우
└── dataset.yaml
```

`dataset.yaml`은 데이터 경로와 클래스 이름을 정의한다.

```yaml
path: C:/ai_course/day15/dataset  # 본인 경로로 변경
train: images/train
val: images/val
test: images/test              # 없으면 생략
names:
  0: helmet
  1: no_helmet
  2: person
```

### 라벨 품질 용어

| 용어 | 의미 / 점검 사항 |
|---|---|
| Occlusion | 다른 물체에 **가려진** 객체; 보이는 부분만/전체 추정 정책 통일 |
| Truncation | 이미지 경계에서 **잘린** 객체; 포함·제외 정책 통일 |
| Crowd | 객체가 밀집한 장면; 개별 박스 분리 기준 |
| Small object | 작은 객체; 최소 크기 기준 |
| Ambiguous class | 클래스가 불명확한 객체; 검토·제외 기준 |
| Negative image | 탐지 대상이 없는 이미지; 거짓 탐지 감소에 도움 |
| Data leakage (데이터 누수) | 같은 원본·영상 구간·유사 프레임이 train/val/test에 섞이는 문제 |
| Split | train(학습) / val(검증·설정 선택) / test(최종 확인) 구분 |
| Label audit | 누락·중복·클래스 범위·좌표 범위·분할 중복 등을 검사 |

**중요:** 이미지 증강본이나 연속 영상 프레임을 서로 다른 split에 무작위로 나누지 않는다. 증강 시 이미지뿐 아니라 **Box도 함께 변환**해야 한다.

## 7. YOLO 모델 구조와 학습 용어

**YOLO (You Only Look Once)**: 한 번의 모델 순전파에서 여러 위치·크기의 객체 후보를 예측하는 **One-stage Detector**.

| 구성/용어 | 역할 |
|---|---|
| Backbone | 이미지의 Edge·Texture·Shape 등 **특징 추출** |
| Neck | 여러 해상도(Scale)의 특징을 **결합** |
| Head | **클래스·박스·신뢰도** 등 탐지 후보 예측 |
| Postprocess | 점수 필터·NMS로 최종 후보 정리(자료의 NMS 사용 경로) |
| Multi-scale | 작은/큰 객체를 위해 여러 크기의 특징 사용 |
| Feature map | 모델이 추출한 이미지 특징 표현 |
| Inference / Predict | 학습된 모델로 새로운 입력을 예측 |
| Training | 정답 라벨로 모델 가중치를 학습 |
| Validation | 검증 데이터로 성능을 확인 |
| Checkpoint | 학습 상태·가중치 저장 지점 |
| Weight (`.pt`) | 저장된 모델 가중치 파일 |

고해상도 특징은 **작은 객체·위치 정보**에, 저해상도 특징은 **큰 객체·넓은 문맥**에 상대적으로 유리하다.

### 학습 설정(Hyperparameters)

| 인자 | 뜻 | 주의할 점 |
|---|---|---|
| `epochs` | 전체 학습 데이터 반복 횟수 | 과소·과적합 |
| `imgsz` | 모델 입력 이미지 크기 | 작은 객체·메모리·속도 |
| `batch` | 한 번에 처리하는 이미지 묶음 | VRAM 사용량 |
| `device` | CPU/GPU 선택 | 실행 환경·속도 |
| `workers` | 데이터 로딩 프로세스 수 | CPU/I/O 병목 |
| `seed` | 재현성을 위한 난수 설정 | 실행 변동 관리 |
| Augmentation | 반전·밝기·크기·Crop·Mosaic·MixUp 등 데이터 변형 | 박스 동시 변환, 현실성 유지 |

### 학습 결과·검증

- `best.pt`: **검증 기준에서 최적인** 체크포인트. 추론·최종 평가의 기본 후보.
- `last.pt`: **마지막으로 저장한** 체크포인트.
- `resume`: 중단된 학습을 재개. 단순히 가중치만 불러오는 것과 구별.
- `results.csv`: Epoch별 Loss·Metric·Learning Rate 등을 확인.
- **Overfitting(과적합):** train은 좋아지지만 val은 정체·악화되는 패턴으로 의심 가능.
- **Images vs Instances:** 검증 이미지 **장수**와 이미지 안의 **객체 개수**는 다르다.

## 8. 추론 결과와 오류 분석

### Ultralytics Results에서 자주 보는 항목

| 항목 | 의미 |
|---|---|
| `result.path` | 입력 파일 경로 |
| `result.orig_shape` | 원본 이미지 크기 `(H,W)` |
| `result.names` | Class ID → 클래스 이름 |
| `result.boxes` | 탐지된 박스들의 집합 |
| `result.plot()` | 박스와 라벨이 그려진 이미지 배열 |
| `result.save()` | 시각화 이미지 저장 |
| `result.to_df()` / `save_txt` | 표 또는 텍스트 형태 결과 활용 |

### 자주 발생하는 탐지 오류

| 오류 | 의미 | 먼저 살펴볼 것 |
|---|---|---|
| Missed object | 실제 객체를 놓침 | 해상도·라벨·Recall |
| False positive | 없는 객체를 탐지 | Negative image·Confidence |
| Wrong class | 객체 종류를 잘못 예측 | 클래스 경계·불균형 |
| Loose box | 박스가 객체에 밀착하지 않음 | 라벨 품질·입력 크기 |
| Duplicate box | 같은 객체에 박스가 여러 개 | NMS IoU·Confidence |
| Domain shift | 야간·비·새 카메라 등 환경 변화로 성능 저하 | 학습/운영 데이터 분포 |

## 9. 다중 객체 추적(Multi-Object Tracking)

```text
Frame 125: person, Track ID 7
Frame 126: person, Track ID 7
Frame 127: person, Track ID 7
```

- **Detection:** 각 프레임에서 객체의 Box·Class·Confidence를 예측.
- **Association:** 이전 프레임과 현재 프레임의 탐지 결과를 연결.
- **Tracking:** 연결 결과에 **Track ID**를 부여·유지.
- **`persist=True`:** 같은 영상의 연속 프레임 호출에서 Tracker 상태를 이어 사용.
- **새 영상/카메라:** 추적 상태를 새로 시작해야 한다.
- **Tracker YAML:** 추적 설정을 관리. Track ID 자체는 추론 중 상태로 유지.

| 추적 오류 | 현상 | 가능한 원인 |
|---|---|---|
| ID switch | 같은 객체의 ID가 바뀜 | 가림·유사 객체·낮은 FPS |
| Track fragmentation | 하나의 이동 경로가 여러 조각으로 끊김 | 탐지 누락·짧은 Buffer |
| False track | 배경에 가짜 ID 생성 | FP·낮은 Confidence |
| Lost track | 가림 이후 다시 연결하지 못함 | Association 설정 |
| Duplicate track | 한 객체에 ID가 여러 개 붙음 | 중복 탐지·NMS |

**핵심:** Tracker를 바꾸는 것만으로 탐지 누락(FN)과 거짓 탐지(FP)가 모두 해결되지는 않는다.

## 10. 속도·운영 지표

| 용어 | 의미 |
|---|---|
| Latency | 프레임 하나를 처리하는 데 걸린 시간 |
| FPS (Frames Per Second) | 1초당 처리 프레임 수 |
| Throughput | 단위 시간당 처리 이미지 수 |
| Peak VRAM | GPU 메모리 최대 사용량 |
| End-to-end 처리 시간 | 영상 읽기(Decode) → 전처리 → 추론 → 추적/후처리 → 저장(Encode) 전체 시간 |

- **모델 추론만 측정한 속도**와 **영상 입출력까지 포함한 FPS**를 혼동하지 않는다.
- 공정한 비교는 입력·가중치·이미지 크기·장치 등을 고정하고 **한 번에 한 설정**만 바꾼다.
- GPU(CUDA)는 비동기 처리이므로 정확한 시간 측정에 **동기화(synchronize)**가 필요할 수 있다.

## 11. 최종 실습 산출물의 의미

| 파일 | 기록 내용 |
|---|---|
| `label_audit.csv` | 클래스·박스·누락·중복·split 검사 |
| `metrics_summary.csv` | Precision·Recall·mAP50·mAP50-95·속도 |
| `detections.csv` | 파일·Box·Class·Confidence |
| `tracking.csv` | Frame·Track ID·Box·Class |
| `demo.mp4` | 박스·ID를 시각화한 영상 |
| `demo_report.md` | 설정·오류 사례·한계·개선안 |
| `README.md` | 환경·실행 명령·재현 방법 |

## 12. 시험·면접 대비 핵심 비교 한눈에 보기

| 헷갈리는 두 개 | 차이 |
|---|---|
| Classification vs Detection | 이미지 전체 분류 vs 객체별 종류·위치 예측 |
| Detection vs Tracking | 한 프레임의 객체 찾기 vs 프레임 사이 동일 객체 연결 |
| Class ID vs Track ID | 객체 **종류** vs 개별 객체의 **시간적 식별** |
| Pixel 좌표 vs Normalized 좌표 | 픽셀 단위 vs 이미지 크기로 나눈 0~1 값 |
| Confidence vs IoU | 예측 후보의 **점수** vs 두 박스의 **겹침** |
| NMS IoU vs 평가 IoU | 예측–예측 중복 제거 vs 예측–정답 매칭 |
| Precision vs Recall | 탐지한 것 중 맞은 비율 vs 실제 객체 중 찾은 비율 |
| AP vs mAP | 한 클래스 PR 요약 vs 클래스별 AP 평균 |
| mAP50 vs mAP50-95 | IoU 0.50 기준 vs 0.50~0.95의 여러 기준 평균 |
| `best.pt` vs `last.pt` | 검증 기준 최적 vs 마지막 저장 |
| Latency vs FPS | 한 프레임 처리 시간 vs 초당 프레임 처리량 |

### 마지막 암기 8문장

1. 객체 탐지는 **클래스 + 위치(Box) + Confidence**를 예측한다.
2. YOLO 라벨은 **`class_id cx cy w h`**, 좌표는 **정규화 값**이다.
3. IoU는 **교집합 / 합집합**이다.
4. NMS는 **중복 예측 박스를 제거**한다.
5. Precision은 **FP**, Recall은 **FN**을 살펴볼 때 중요하다.
6. mAP50-95는 mAP50보다 **다양하고 엄격한 위치 기준**을 반영한다.
7. YOLO는 **Backbone(추출) → Neck(결합) → Head(예측)** 구조로 이해한다.
8. Tracking은 **연속 프레임의 같은 객체에 Track ID를 유지**하는 작업이다.

---

**자료 기준 안내:** 이 문서는 업로드한 Day 15 강의 자료의 개념과 용어를 학습용으로 재구성한 요약이다. 코드 실행 결과나 실제 모델 성능을 측정한 문서는 아니다. 자료의 예시 환경은 **Ultralytics 8.4.142 / YOLO26n** 기준이며, 설치·실행 설정은 사용하는 환경에 맞춰 확인해야 한다.
