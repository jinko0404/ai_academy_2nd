"""실습 07~08 응용: 얕은 폴더 구조에서 분류 실험표를 완성합니다."""
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
# 모델에 입력하기 전 데이터를 정리하는 전처리 규칙을 정의한다.
# 여기서는 전처리 방법만 설정하며, 실제 데이터에 맞춘 학습은 fit할 때 진행
pre = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), FEATURES[:2]),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), FEATURES[2:]),
])
# 트리는 이 실습에서 표준화하지 않습니다. 각 후보·fold마다 새 전처리기를 학습합니다.
# 후보 모델과 각 fold마다 전처리기를 새로 학습한다.

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, confusion_matrix

# train 데이터를 5개 fold로 나눌 규칙을 만들고 실제 분할 결과를 리스트로 저장
folds = list(
    StratifiedKFold(
        n_splits=5,     # 데이터를 5묶음으로 분할
        shuffle=True,   # 데이터를 무작위로 섞은 뒤 분할
        random_state=42 # 실행할 때 마다 같은 분할 결과가 나오도록 한다.
        ).split(X_train, y_train)
    )


# 비교할 4개 분류 모델과 각 모델 설정
candidates = [
    # 기준 모델: 학습 데이터에서 가장 많이 등장하는 클래스만 예측한다.
    ("Dummy", DummyClassifier(strategy="most_frequent")),

    # 결정 트리: 하나의 트리를 이용해 분류한다.
    ("Tree", DecisionTreeClassifier(
        max_depth=3,        # 트리의 최대 깊이를 3으로 제한 
        min_samples_leaf=3, # 끝 노드에 필요한 최소한의 데이터 수 3개
        random_state=42     # 같은 조건에서 무작위 요소가 그대로 재현 되도록 설정
        )), 

    # 랜덤 포레스트 : 여러 결정 트리의 예측을 결합
    ("Forest", RandomForestClassifier(
        n_estimators=100,       # 트리 100개 사용
        max_depth=4,            # 각 트리별 최대 깊이 4로 제한
        min_samples_leaf=2,     # 각 트리의 끝 노드에 필요한 최소 데이터 수 2개
        max_features="sqrt",    # 분기할 때 고려하는 특성 수를 전체 특성 수의 제곱근으로 설정
        random_state=42,        # 같은 조건에서 무작위 요소가 그대로 재현 되도록 설정
        n_jobs=1                # 학습 작업에 CPU 작업자 1개 사용
        )), 

    # XGBoost: 여러 트리를 순차적으로 학습하며 앞선 예측의 오류를 보완한다.
    ("XGB", XGBClassifier(
        n_estimators=80,                    # 트리 80개 사용
        max_depth=2,                        # 각 트리별 최대 깊이 2로 제한
        learning_rate=0.05,                 # 학습률 5프로
        subsample=0.8,                      # 각 트리 학습마다 전체 학습 데이터의 80% 사용한다.
        colsample_bytree=1.0,               # 각 트리에서 전체 특성을 사용한다.
        reg_lambda=1.0,                     # L2 규제 강도를 1.0으로 설정
        objective="binary:logistic",        # 이진 분류를 위한 확률 예측 방식 사용
        eval_metric="logloss",              # 학습 중 사용할 평가 지표로 Log Loss 지정
        tree_method="hist",                 # 히스토그램 방식으로 트리를 학습합니다.
        device="cpu",                       # CPU로 학습한다.
        random_state=42,                    # 같은 조건에서 무작위 요소가 그대로 재현 되도록 설정
        n_jobs=1                            # 학습 작업에 CPU 작업자 1개 사용
        ))
    ]
# 모델별 교차검증 결과와 상세 점수를 저장할 빈 리스트
rows, cv_rows = [], []

# valid 데이터의 ID와 실제 정답을 저장할 표를 만듭니다.
# 이후 각 모델의 예측 결과를 열로 추가합니다.
predictions = pd.DataFrame({"request_id": X_valid.index, "actual": y_valid.to_numpy()})

# 후보 모델을 하나씩 꺼내서 평가
# name에는 모델 이름, estimator에는 해당 모델 객체가 들어갑니다.
for name, estimator in candidates:

    #전처리와 모델을 하나의 파이프라인으로 연결
    # clone(pre)는 전처리 설정을 복사하되 학습된 정보는 복사하지 않습니다.
    model = Pipeline([("pre", clone(pre)), ("model", estimator)])
    # TODO A07a: train 내부 교차검증 뒤 전체 train으로 학습하세요.
    # train을 5개 fold로 나누어 각 fold의 F1 점수를 계산합니다.
    # 매번 4개 fold로 학습하고 나머지 1개 fold로 평가합니다.
    scores = cross_val_score(
            model,          # 전처리와 분류 모델을 연결한 파이프라인
            X_train,        # 학습용 입력 데이터
            y_train,        # 학습용 정답 데이터
            cv=folds,       # 앞에서 만든 동일한 5개 분할 사용(43~50라인 참고)
            scoring="f1",   # F1 점수로 모델 성능 평가
            n_jobs=1        # CPU 작업을 하나씩 진행
        )
    model.fit(X_train, y_train)
    # TODO 끝
    # TODO A07b: valid의 예측을 prediction에 넣으세요.
    prediction = model.predict(X_valid)# cross_val_score의 fold별 학습과는 별개의 학습
    # TODO 끝

    # 실제 정답과 예측을 비교해 혼동 행렬을 계산합니다.
    # labels=[0, 1]은 정상(0), 실패(1)의 순서를 지정합니다.
    # ravel()로 TN, FP, FN, TP 네 값을 각각 꺼냅니다.
    tn, fp, fn, tp = confusion_matrix(y_valid, prediction, labels=[0, 1]).ravel()

    # 현재 모델의 주요 평가 결과를 한 행으로 저장합니다.
    rows.append({
        "model": name,          # 모델 이름
        "CV_F1": scores.mean(), # 5개 fold의 F1 점수 평균
        "CV_std": scores.std(), # 5개 fold의 F1 점수의 표준 편차
        "train_F1": f1_score(
            y_train, 
            model.predict(X_train), 
            zero_division=0
        ), # 학습에 사용한 train 데이터에 대한 F1
        "valid_F1": f1_score(
            y_valid, 
            prediction, 
            zero_division=0
        ), # 별도의 valid 데이터에 대한 F1
        "FP": fp,   # 실제 정상(0)이지만 실패(1)로 잘못 예측한 건수
        "FN": fn    # 실제 실패(1)이지만 정상(0)으로 잘못 예측한 건수
    })

    # 각 fold의 점수를 개별 행으로 저장합니다.
    for number, value in enumerate(scores, start=1):
        cv_rows.append({
            "model": name,      # 모델 이름
            "fold": number,     # fold 번호
            "fit_n": 48,        # 각 fold의 학습 데이터 수
            "evaluate_n": 12,   # 각 fold의 평가 데이터 수
            "F1": value         # 해당 fold의 F1 점수
        })

    # 현재 모델의 valid 예측을 predictions 표에 새 열로 추가합니다.
    predictions[name] = prediction

    # 랜덤 포레스트가 학습된 경우 특성 중요도를 계산해 표로 만듭니다.
    if name == "Forest":
        importance = pd.DataFrame({
            # 전처리 후 실제 모델이 사용하는 특성 이름
            "feature": model["pre"].get_feature_names_out(), 

            # 랜덤 포레스트가 계산한 각 특성의 중요도
            "importance": model["model"].feature_importances_
        })

comparison = pd.DataFrame(rows) # # 모델별 요약 결과를 DataFrame으로 변환합니다.

# 모델별 교차검증 평균, 표준편차, train/valid F1, FP/FN을 저장합니다.
comparison.to_csv(OUT / "classification_comparison.csv", index=False, encoding="utf-8-sig")

# 모델별 각 fold의 F1 점수를 저장합니다.
pd.DataFrame(cv_rows).to_csv(OUT / "cv_scores.csv", index=False, encoding="utf-8-sig")

# valid 데이터의 실제 정답과 각 모델의 예측 결과를 저장합니다.
predictions.to_csv(OUT / "validation_predictions.csv", index=False, encoding="utf-8-sig")

# 랜덤 포레스트의 특성 중요도를 저장합니다.
importance.to_csv(OUT / "feature_importance.csv", index=False, encoding="utf-8-sig")

# 모델별 평가 결과를 소수점 넷째 자리까지 출력
print(comparison.round(4).to_string(index=False))

# CV 평균 F1이 가장 높은 모델을 선택 기준으로 삼습니다.
# 점수가 같으면 candidates에 미리 적힌 순서를 따릅니다.
# valid 점수는 최종 선택 기준과 별도로 확인하는 진단 자료입니다.
print("CV 평균 F1 최대, 동률이면 미리 제시한 후보 순서로 선택합니다. valid는 별도 진단입니다.")
print("저장: outputs/classification_comparison.csv, cv_scores.csv, validation_predictions.csv, feature_importance.csv")
print("test는 학습·변환·예측·평가하지 않았습니다.")
