"""실습 03 응용: 같은 조건의 단일 트리와 숲을 비교합니다."""
from pathlib import Path
import pandas as pd

# 제공: 전날의 분할 소속뿐 아니라 position 순서까지 그대로 읽습니다.
# 파일 경로 설정 및 읽기
HERE = Path(__file__).resolve().parent
raw = pd.read_csv(HERE / "data/requests.csv").set_index("request_id")
manifest = pd.read_csv(HERE / "data/split_manifest.csv")

# 읽어들인 정보 토대로 데이터 셋팅
train_ids = manifest.loc[manifest["split"] == "train"].sort_values("position")["request_id"]
valid_ids = manifest.loc[manifest["split"] == "valid"].sort_values("position")["request_id"]
train = raw.loc[train_ids]
valid = raw.loc[valid_ids]
FEATURES = ["input_chars", "allowed_tools", "task_type", "priority"]
X_train, X_valid = train[FEATURES], valid[FEATURES]
y_train, y_valid = train["failed"], valid["failed"]

# 출력 파일 저장할 폴더 경로 설정
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
# test 20건의 입력·정답은 아래 모델에 전달하지 않습니다.
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone

# 제공: 전날 배운 결측 대체와 문자 변환입니다. 통계는 fit할 때만 배웁니다.
# 숫자형 데이터와 문자형 데이터에 서로 다른 전처리를 적용하도록 설정
pre = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), FEATURES[:2]),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), FEATURES[2:]),
])
# 트리는 이 실습에서 표준화하지 않습니다. 각 후보·fold마다 새 전처리기를 학습합니다.

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix

# 비교할 모델 2가지를 목록으로 생성
candidates = [
    #첫번째 모델 결정 트리 하나(트리 최대 깊이 3, 리프 노드마다 3개의 학습 샘플 요구, 실행결과를 재현 하도록 설정)
    ("Tree", DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, random_state=42)),
    #두번째 모델 랜덤 포레스트(결정 트리 100개 사용, 각 트리 최대 깊이 4, 리프 노드마다 최소 2개의 학습 샘플 요구
    #                       각 분기에서 고려할 특성 수를 제곱근 기준으로 설정, 실행 결과를 재현할 수 있도록 설정,
    #                       한 번에 작업할 CPU 작업자 수를 1로 설정)
    ("Forest", RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=2,
                                    max_features="sqrt", random_state=42, n_jobs=1))
                                    ]

rows = []# 각 모델의 평가 결과를 저장할 빈 리스트

# candidates에서 모델 이름과 모델 설정을 하나씩 꺼내 반복
for name, estimator in candidates:
    # 전처리 과정과 모델을 하나의 파이프라인으로 연결
    # clone(pre)는 전처리 설정을 복사해 새로운 전처리기를 만듦
    model = Pipeline([("pre", clone(pre)), ("model", estimator)])
    # TODO A03: model을 train으로 학습한 뒤 valid를 예측하세요.
    model.fit(X_train, y_train)
    prediction = model.predict(X_valid)
    # TODO 끝

    # 실제 정답과 예측 결과를 비교해 혼동 행렬을 계산
    # labels=[0, 1]은 정답과 예측의 클래스 순서를 0, 1로 고정
    # ravel()은 2x2 결과를 tn, fp, fn, tp 네 값으로 펼침
    tn, fp, fn, tp = confusion_matrix(y_valid, prediction, labels=[0, 1]).ravel()

    # 현재 모델의 평가 결과를 딕셔너리로 만들어 리스트에 추가
    #모델 이름, (각각 학습, 검증 데이터에 대한 예측과 실제 정답을 비교한 F1 점수)
    #실제로는 정상(0)이지만 실패(1)라고 잘못 예측한 개수와 실제로는 실패(1)이지만 정상(0)라고 잘못 예측한 개수
    rows.append({"model": name, "train_F1": f1_score(y_train, model.predict(X_train), zero_division=0),
                 "valid_F1": f1_score(y_valid, prediction, zero_division=0), "FP": fp, "FN": fn})

result = pd.DataFrame(rows)# 모델별 평가 결과를 판다스 DataFrame으로 변환
result.to_csv(OUT / "03_forest.csv", index=False, encoding="utf-8-sig")# 평가 결과를 CSV 파일로 저장

# 점수를 소수점 셋째 자리까지 반올림하여 표 형태로 출력
print(result.round(3).to_string(index=False))
