"""실습 05 응용: PPT와 같은 XGBoost 80회차 분류 모델입니다."""
from pathlib import Path
import pandas as pd

# 제공: 전날의 분할 소속뿐 아니라 position 순서까지 그대로 읽습니다.
HERE = Path(__file__).resolve().parent
raw = pd.read_csv(HERE / "data/requests.csv").set_index("request_id")
manifest = pd.read_csv(HERE / "data/split_manifest.csv")
train_ids = manifest.loc[manifest["split"] == "train"].sort_values("position")["request_id"]
valid_ids = manifest.loc[manifest["split"] == "valid"].sort_values("position")["request_id"]
train = raw.loc[train_ids]
valid = raw.loc[valid_ids]
FEATURES = ["input_chars", "allowed_tools", "task_type", "priority"]
X_train, X_valid = train[FEATURES], valid[FEATURES]
y_train, y_valid = train["failed"], valid["failed"]
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
# test 20건의 입력·정답은 아래 모델에 전달하지 않습니다.
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone

# 제공: 전날 배운 결측 대체와 문자 변환입니다. 통계는 fit할 때만 배웁니다.
# 각각의 숫자 데이터와 문자 데이터 전처리 과정 구성
pre = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), FEATURES[:2]),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), FEATURES[2:]),
])
# 트리는 이 실습에서 표준화하지 않습니다. 각 후보·fold마다 새 전처리기를 학습합니다.

from xgboost import XGBClassifier
from sklearn.metrics import f1_score, confusion_matrix, log_loss

# XGBoost 모델의 설정을 지정
estimator = XGBClassifier(
    n_estimators=80,                # 트리를 총 80 회차 학습
    max_depth=2,                    # 각 트리 별 최대 깊이 2로 제한
    learning_rate=0.05,             # 각 트리의 학습 기여도 조절
    subsample=0.8,                  # 각 트리 학습에서 학습 데이터 80% 사용
    colsample_bytree=1.0,           # 각 트리에서 전체 특성을 사용할 수 있다.
    reg_lambda=1.0,                 # 과적합을 줄이는 L2 규제 강도
    objective="binary:logistic",    # 이진 분류 확률을 예측하도록 설정
    eval_metric="logloss",          # 학습 과정에서 사용할 평가 지표
    tree_method="hist",             # 히스토그램 기반 트리 학습 방식 사용
    device="cpu",                   # CPU로 학습
    random_state=42,                # 난수 결과를 재현하도록 설정
    n_jobs=1                        # 작업에 CPU 프로젝트 1개 사용
                  )

#전처리 과정과 XGBoost 모델을 하나의 파이프라인으로 연결한다.
model = Pipeline([("pre", clone(pre)), ("model", estimator)])

# TODO A05: Pipeline을 학습하고 valid를 예측하세요.
model.fit(X_train, y_train) # 파이프 라인에 학습데이터 학습
prediction = model.predict(X_valid) #학습데이터 기반으로 valid 값 예측
# TODO 끝

# 각각의 검증 데이터에 대해 실패 클래스(1)에 속할 확률
probability = model.predict_proba(X_valid)[:, 1]  # classes_ = [0, 1] 확인 후 실패1의 열 사용

# 모델이 인식하는 클래스 순서
print("클래스 순서:", model.classes_.tolist())

# 실제 정답과 예측 결과를 비교해 혼동 행렬을 계산합니다.
# labels=[0, 1]로 클래스 순서를 명시합니다.
# ravel()로 행렬의 네 값을 순서대로 꺼냅니다.
tn, fp, fn, tp = confusion_matrix(y_valid, prediction, labels=[0, 1]).ravel()

# 성능 평가 결과를 표로 저장한다.
result = pd.DataFrame([{"train_F1": f1_score(y_train, model.predict(X_train), zero_division=0),
                       "valid_F1": f1_score(y_valid, prediction, zero_division=0), "FP": fp, "FN": fn,
                       "valid_logloss": log_loss(y_valid, probability, labels=[0, 1])}])

#표로 저장된 결과를 csv형태로 저장 및 출력
result.to_csv(OUT / "05_xgboost.csv", index=False, encoding="utf-8-sig")
print(result.round(3).to_string(index=False))
print("objective는 학습 목적, logloss는 진단용, 후보 비교의 주 지표는 F1입니다.")
