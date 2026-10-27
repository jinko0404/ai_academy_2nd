"""실습 06 응용: Pipeline 전체를 같은 train 5-fold로 평가합니다."""
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
pre = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), FEATURES[:2]),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), FEATURES[2:]),
])
# 트리는 이 실습에서 표준화하지 않습니다. 각 후보·fold마다 새 전처리기를 학습합니다.

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# train 데이터 60개를 5개 fold로 나누는 규칙
folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_train, y_train))

#비교할 분류 모델 3가지와 각 모델 설정
candidates = [
    # 결정 트리 모델
    ("Tree", DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, random_state=42)),

    #여러 결정 트리를 결합한 랜덤 포레스트
    ("Forest", RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=2,
                                    max_features="sqrt", random_state=42, n_jobs=1)),

    #여러 트리를 순차적으로 학습한 XGBoost
    ("XGB", XGBClassifier(n_estimators=80, max_depth=2, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=1.0, reg_lambda=1.0,
                  objective="binary:logistic", eval_metric="logloss",
                  tree_method="hist", device="cpu", random_state=42, n_jobs=1))
]

rows = [] # 모델별 평가를 저장할 빈 리스트

#후보 모델을 꺼내서 하나씩 평가
for name, estimator in candidates:

    #전처리와 모델을 하나의 파이프 라인으로 연결
    model = Pipeline([("pre", clone(pre)), ("model", estimator)])

    #5개의 fold로 각각 학습하고 F1 점수 계산
    # TODO A06: 전처리까지 포함한 model의 fold별 F1을 계산하세요.
    scores = cross_val_score(
        model,          # 전처리 + 모델 전체
        X_train,        # 학습용 데이터
        y_train,        # 학습용 정답
        cv=folds,       # 앞에서 만든 5개 분할 사용(39 라인 참조)
        scoring="f1",   # 평가 기준을 f1으로 설정
        n_jobs=1        # 작업에 사용될 cpu 프로세스 1개
    )
    # TODO 끝

    #평가결과 출력 및 DataFrame으로 바꾸고 CSV 파일로 저장
    print(name, "fold F1:", scores.round(4).tolist(), "평균:", round(scores.mean(), 4), "표준편차:", round(scores.std(), 4))
    for number, value in enumerate(scores, start=1):
        rows.append({"model": name, "fold": number, "fit_n": 48, "evaluate_n": 12, "F1": value})
pd.DataFrame(rows).to_csv(OUT / "06_cv_scores.csv", index=False, encoding="utf-8-sig")
print("사용 자료: train60 내부만 / valid20·test20 미사용")
