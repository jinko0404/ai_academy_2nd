"""실습 03 응용: Pipeline을 통째로 train 내부 5-fold에 넣어 다중 지표를 계산합니다."""
from pathlib import Path
import pandas as pd
import numpy as np

HERE = Path(__file__).resolve().parent
raw = pd.read_csv(HERE / "data/requests.csv").set_index("request_id")
manifest = pd.read_csv(HERE / "data/split_manifest.csv")
train_ids = manifest.loc[manifest["split"] == "train"].sort_values("position")["request_id"].tolist()
valid_ids = manifest.loc[manifest["split"] == "valid"].sort_values("position")["request_id"].tolist()
train, valid = raw.loc[train_ids].copy(), raw.loc[valid_ids].copy()

FEATURES = ["input_chars", "allowed_tools", "task_type", "priority"]
NUM, CAT = FEATURES[:2], FEATURES[2:]
X_train, X_valid = train[FEATURES].copy(), valid[FEATURES].copy()
y_train, y_valid = train["failed"], valid["failed"]

OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# 제공: 모든 평균·중앙값·범주 목록은 Pipeline.fit의 학습 부분에서만 결정됩니다.
# 각 숫자형, 범주형 데이터 서로 다른 전처리 방식 설정
pre = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                      ("scaler", StandardScaler())]), NUM),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                      ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CAT),
])

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, f1_score

# 전처리 과정과 로지틱스 회귀 모델을 하나의 파이프라인으로 연결
model = Pipeline([("pre", pre), ("model", LogisticRegression(C=1, max_iter=1000))])

# 5-Fold 교차검증 방식 설정
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# TODO A03: train만 넣어 F1과 AUC를 함께 구하세요. 전처리한 배열 대신 model 전체를 전달합니다.
# train 데이터만 사용하여 5-Fold 교차검증 실행
# 각 fold에서 Pipeline의 전처리부터 모델 학습까지 다시 수행
# 평가 지표로 F1과 ROC-AUC를 함께 계산
scores = cross_validate(
    model,                     # 전처리 + 모델 학습 전체 과정(40 라인 참고)
    X_train,                   # train 데이터의 입력 특성
    y_train,                   # train 데이터의 정답
    cv=folds,                  # 위에서 설정한 5-Fold 방식(43라인 참고)
    scoring=["f1", "roc_auc"]  # 계산할 평가 지표 두 가지(F1, AUC)
)
# TODO 끝
# 각 fold의 결과를 표로 정리
result = pd.DataFrame({
    "fold": range(1, 6),                # 1~5번 fold
    "F1": scores["test_f1"],            # 각 fold의 F1 점수
    "ROC_AUC": scores["test_roc_auc"]   # 각 fold의 ROC-AUC 점수
})
result.to_csv(OUT / "classification_cv.csv", index=False, encoding="utf-8-sig")
print(result.round(3).to_string(index=False))
print("평균F1/표준편차:", round(result["F1"].mean(), 3), round(result["F1"].std(ddof=0), 3))
print("각 fold: 학습48·평가12. test_f1의 test는 CV 내부 평가이며 보관 test20이 아닙니다.")
