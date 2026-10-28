"""실습 03 심화: 회귀 Pipeline 두 개를 같은 KFold로 비교합니다."""
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

# 숫자형, 범주형 데이터 따라 전처리 방식 다르게 설정
pre = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                      ("scaler", StandardScaler())]), NUM),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                      ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CAT),
])

from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold, cross_val_score

# 데이터를 5개 묶음으로 나누는 교차검증 방식 설정
# shuffle=True: 데이터를 나누기 전에 섞음
# random_state=42: 실행할 때마다 동일한 방식으로 섞이도록 고정
# split(X_train): 입력 데이터를 5개 fold로 나누는 인덱스를 생성
# list(): 생성된 분할 정보를 리스트에 저장
folds = list(KFold(n_splits=5, shuffle=True, random_state=42).split(X_train))

rows = [] # 각 모델의 fold별 RMSE 결과를 저장할 빈 리스트

# 두 가지 회귀 모델을 차례대로 비교
# name: 결과표에 기록할 모델 이름
# estimator: 실제로 사용할 회귀 모델
for name, estimator in [(
    "Linear", LinearRegression()),                  # 선형 회귀 모델
    ("KNN15", KNeighborsRegressor(n_neighbors=15)   # 이웃 15개를 사용하는 KNN 회귀 모델
    )]:
    # 전처리 과정과 현재 회귀 모델을 하나의 Pipeline으로 연결
    # clone(pre): 전처리 설정을 복사해 모델마다 독립적으로 사용
    # 전처리 기준은 각 fold의 학습 데이터로만 학습됨
    model = Pipeline([("pre", clone(pre)), ("model", estimator)])

    # TODO H03: 같은 folds로 음수 RMSE 점수를 구한 뒤 부호를 되돌리세요.
    # 동일한 5개 fold로 교차검증을 수행하고 RMSE 계산
    # neg_root_mean_squared_error: RMSE에 음수 부호를 붙인 평가 점수
    # sklearn은 평가 점수를 클수록 좋게 통일하기 때문에 음수로 반환
    # 맨 앞의 -를 붙여 실제 RMSE인 양수로 변환
    rmse = -cross_val_score(
        model,                              # 전처리 + 회귀 모델 전체
        X_train,                            # 학습용 입력 데이터
        y_train,                            # 실제 정답값(elapsed_ms)
        cv=folds,                           # 앞에서 만든 동일한 5개 데이터 분할
        scoring="neg_root_mean_squared_error"  # 음수 RMSE 평가 방식
    )
    # TODO 끝

    # 5개 fold의 RMSE를 하나씩 꺼내 결과 리스트에 저장
    # enumerate(..., start=1): fold 번호를 1부터 시작
    for number, value in enumerate(rmse, start=1):
        # 모델 이름, fold 번호, RMSE 값을 딕셔너리로 저장
        rows.append({
            "model": name,      # 모델 이름(Linear 또는 KNN15)
            "fold": number,     # 교차검증 묶음 번호
            "RMSE_ms": value     # 해당 fold의 RMSE(ms)
        })

# 결과 리스트 표로 변환하여 출력 및 파일 저장
result = pd.DataFrame(rows)
result.to_csv(OUT / "regression_cv.csv", index=False, encoding="utf-8-sig")
print(result.round(2).to_string(index=False))
print("평균 RMSE(ms)")
print(result.groupby("model", sort=False)["RMSE_ms"].mean().round(2).to_string())
print("회귀 교차검증의 정답은 elapsed_ms입니다. 입력에 elapsed_ms를 포함하지 않습니다.")
