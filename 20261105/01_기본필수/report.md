# Day14 기본필수 · 전이학습 프로젝트 보고서

> 사전학습 Backbone을 활용한 Head-only 이미지 분류

## 1. 프로젝트 목적

사전학습한 **Backbone을 고정하고 Head만 학습**하여 이미지를 세 클래스로 분류한다. 검증 데이터로 최적 epoch를 선택하고, 검증 성능과 이미지별 예측 결과, Grad-CAM을 기록한다.

## 2. 데이터 구성과 분할 확인

### 클래스 매핑 검사

`ImageFolder`가 생성한 클래스 이름과 번호의 매핑은 다음과 같다.

```python
{'circle': 0, 'square': 1, 'triangle': 2}
```

| 데이터 분할 | 기준 train과 클래스 매핑 일치 여부 |
|---|:---:|
| train | 일치 |
| valid | 일치 |
| test | 일치 |

### 클래스별 이미지 수와 비율

| 데이터 분할 | 클래스 | 이미지 수 | 분할 내 비율 |
|---|---|---:|---:|
| train | circle | 12 | 57.14% |
| train | square | 6 | 28.57% |
| train | triangle | 3 | 14.29% |
| valid | circle | 3 | 33.33% |
| valid | square | 3 | 33.33% |
| valid | triangle | 3 | 33.33% |
| test | circle | 2 | 33.33% |
| test | square | 2 | 33.33% |
| test | triangle | 2 | 33.33% |

**전체 이미지 수:** train 21장 · valid 9장 · test 6장, 총 36장

train에서는 circle 클래스가 가장 많아 클래스 불균형이 존재했다. valid는 클래스마다 3장으로 균일했으며, train·valid·test의 클래스 번호 매핑은 모두 일치했다.

### 데이터 사용 범위

- **train:** Head의 파라미터 학습에 사용했다.
- **valid:** epoch별 검증 손실을 비교하여 최적 epoch를 선택하고, 최종 검증 지표와 예측 결과를 계산했다. 검증 손실 이력은 [history.csv](history.csv)에 저장했다.
- **test:** 사전 데이터 점검에서 개수와 클래스 매핑만 확인했으며, `07_project.py`의 학습·추론·Grad-CAM에는 사용하지 않았다.

## 3. 사전학습 가중치와 입력 규칙

| 기록 항목 | 적용 내용 |
|---|---|
| 모델 구조 | 소형 CNN인 TinyShapeCNN. 합성곱층 2개와 BatchNorm, ReLU, Pooling, Flatten, Linear Head로 구성 |
| source 출처 | 별도 합성 이미지로 사전학습한 가중치 사용. ImageNet ResNet18 가중치가 아님 |
| 불러온 가중치 파일 | `data/source_backbone.npz` |
| 입력 크기 | `Resize((32, 32))`로 높이·너비를 32×32로 변환 |
| 색상 채널 | RGB 3채널 |
| 픽셀 범위 | `ToTensor()`로 일반적인 8비트 이미지의 0~255 값을 실수 `[0, 1]`로 변환 |
| 입력 Tensor 형태 | `(배치 크기, 3, 32, 32)` |
| Head의 입력·출력 | `nn.Linear(256, 3)`. 256개 특징을 받아 클래스별 logit 3개 출력 |
| 입력 정규화 | ImageNet의 mean/std 정규화는 적용하지 않음 |

## 4. 학습 범위와 실험 설정

| 항목 | 적용 내용 |
|---|---|
| 학습하는 층 | Head인 `model[9]`의 가중치와 편향만 학습. 총 771개 파라미터 |
| 고정하는 층 | Backbone인 `model[0]`~`model[8]`의 파라미터를 `requires_grad=False`로 고정 |
| BatchNorm 통계 정책 | `model.eval()`로 이동 평균·분산을 고정. Head의 기울기 계산과 학습은 유지 |
| optimizer와 학습률 | AdamW 사용. `lr=0.01`, `weight_decay=0.0` |
| 손실 함수 | `nn.CrossEntropyLoss()`. 클래스 가중치를 적용하지 않음 |
| 최대 학습 횟수 | 8 epoch. 각 epoch마다 전체 train 데이터를 한 번에 사용 |
| 난수 seed | `torch.manual_seed(14)`로 PyTorch 난수 시드 고정 |
| 최적 epoch 선택 기준 | `valid_loss < best_loss`일 때 모델 상태를 저장하고, 학습 후 검증 손실이 가장 낮았던 상태를 복원 |

## 5. 산출물

결과 파일의 저장 위치는 `01_기본필수` 폴더이다.

| 결과 파일 | 기록 내용 |
|---|---|
| [metrics.csv](metrics.csv) / [metrics.json](metrics.json) | Accuracy, Macro F1, 최적 epoch, 검증 이미지 수 등 |
| [history.csv](history.csv) | epoch별 검증 손실 변화 |
| [predictions.csv](predictions.csv) | 이미지별 파일 이름, 정답, 예측, 예측 클래스의 확률 |
| [confusion.csv](confusion.csv) | 실제 클래스와 예측 클래스별 개수. 행은 실제, 열은 예측 |
| `project_cam.png` | 선택한 이미지의 예측 클래스를 기준으로 계산한 공간적 반응 지도 |

> Grad-CAM 이미지는 실제 저장 여부를 확인한 뒤 보고서에 첨부한다. 지도는 선택한 점수에 대한 반응을 보여주며 예측 원인이나 정답을 증명하지 않는다.

## 6. 오답 분석과 한계

| 구분 | 내용 |
|---|---|
| 관찰 | 최종 모델의 검증 데이터에서 오답이 0건이었다. 따라서 분석할 오답 이미지가 없으며, 개별 오답의 실제 클래스·예측 클래스·확률 기록은 해당하지 않는다. |
| 가설 | 오답 사례가 없어 오분류 원인에 대한 가설은 제시하지 않았다. |
| 한계 | 이번 검증 데이터에서 모두 맞혔다는 의미이며, 새로운 데이터에서도 항상 정확하다고 단정할 수 없다. valid를 최적 epoch 선택에도 사용했으므로 독립적인 최종 평가 결과는 아니다. |
| 개선 사항 | 향후 실험에서는 더 많고 다양한 데이터를 별도로 확보하여 일반화 성능을 확인한다. 오답이 발생하면 실제 이미지와 예측 결과를 근거로 원인을 분석한다. |

05·06번 실습의 초기 모델에서 관찰한 오류는 이번 07번 최종 모델의 결과와 구분하여 해석한다.
