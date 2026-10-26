"""실습 07~08 응용: 한 파일에서 회귀·분류 각각 세 후보를 비교합니다."""
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
raw = pd.read_csv(HERE / "data/requests.csv").set_index("request_id")
# 이 표는 전날의 분할 소속과 각 분할 내부의 행 순서(position)를 함께 보존합니다.
split = pd.read_csv(HERE / "data/split_manifest.csv")

# 요청 데이터의 아이디를 리스트 형태로 저장
train_ids = split.loc[split["split"] == "train"].sort_values("position")["request_id"]
valid_ids = split.loc[split["split"] == "valid"].sort_values("position")["request_id"]

#11, 12라인에서 저장한 리스트를 활용하여 데이터 추출
train = raw.loc[train_ids]
valid = raw.loc[valid_ids]

# 입력 데이터 종류 설정
features = ["input_chars", "allowed_tools", "task_type", "priority"]
X_train = train[features]
X_valid = valid[features]
# test의 ID는 분할표에만 남겨 둡니다. test 입력·정답을 모델에 전달하지 않습니다.
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, precision_score, recall_score, f1_score, confusion_matrix

# 제공: 전날 만든 전처리입니다. 숫자 2열은 중앙값→표준화, 문자 2열은 최빈값→원핫입니다.
num = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
pre = ColumnTransformer([("num", num, features[:2]), ("cat", cat, features[2:])])

# 제공 K: PPT 실습06의 동일 원본·분할·standard 탐색 결과(분류3, 회귀15)를 재사용합니다.
regression_rows = [] #회귀 모델의 평가 결과 저장할 리스트(58~60라인 참고)
classification_rows = [] #분류 모델의 평가 결과 저장할 리스트(88~94라인 참고)
predictions = [] # 각 모델의 실제 정답과 예측 결과를 저장할 리스트(63~64라인, 97~98라인 참고)

# 회귀 평가모델 Dummy, Linear, KNN15 3가지 비교
# Dummy = 기준 모델, Linear = 선형 회귀, KNN15 = 이웃 15개를 사용하는 KNN 회귀
for name, estimator in [("Dummy", DummyRegressor()), ("Linear", LinearRegression()),
                        ("KNN15", KNeighborsRegressor(n_neighbors=15))]:
    # 기존 전처리 과정과 회귀 모델을 하나의 파이프라인으로 구성한다.
    model = Pipeline([("pre", clone(pre)), ("model", estimator)])

    # TODO A07r: 회귀 모델을 학습하고 valid 예측을 구하세요.
    # 힌트: 정답 열은 elapsed_ms입니다.
    model.fit(X_train, train["elapsed_ms"])# 학습 데이터의 실행 시간(elapsed_ms)을 정답으로 활용해 모델을 학습한다.
    prediction = model.predict(X_valid)# 검증 데이터의 실행 시간(elapsed_ms) 예측

    #실제 실행 시간과 예측 시간을 비교하여 MAE와 TMSE 계산 후 기록(41라인 참고)
    regression_rows.append({"model": name, "scale": "standard",
        "MAE_ms": mean_absolute_error(valid["elapsed_ms"], prediction),
        "RMSE_ms": root_mean_squared_error(valid["elapsed_ms"], prediction)})

    #실제 정답과 예측 결과로 표로 구성하여 리스트에 추가한다.(43 라인 참고)
    predictions.append(pd.DataFrame({"request_id": valid.index, "task": "regression", "model": name,
        "actual": valid["elapsed_ms"].to_numpy(), "predicted": prediction}))

#분류 평가 모델 Dummy, Logistic, KNN3 3가지 비교
#Dummy = 가장 자주 등장하는 정답 예측, Logistic = 로지스틱 회귀, KNN3 = 이웃 3개를 활용한 KNN 분류
for name, estimator in [("Dummy", DummyClassifier(strategy="most_frequent")),
                        ("Logistic", LogisticRegression(C=1.0, max_iter=1000)),
                        ("KNN3", KNeighborsClassifier(n_neighbors=3))]:

    # 기존 전처리 과정과 분류 모델을 하나의 파이프라인으로 구성한다.
    model = Pipeline([("pre", clone(pre)), ("model", estimator)])

    # TODO A07c: 분류 모델을 학습하고 valid 예측을 구하세요.
    # 힌트: 정답 열은 failed입니다.
    model.fit(X_train, train["failed"]) #학습 데이터의 실패 여부(failed)를 정답으로 활용하여 모델 학습
    prediction = model.predict(X_valid) # 검증 데이터의 실패 여부(failed)를 예측한다.

    # 실제 정답과 예측 결과를 비교해 혼동 행렬의 값을 계산합니다.
    # tn: 정상 요청을 정상으로 예측
    # fp: 정상 요청을 실패로 잘못 예측
    # fn: 실패 요청을 정상으로 잘못 예측
    # tp: 실패 요청을 실패로 정확히 예측
    tn, fp, fn, tp = confusion_matrix(valid["failed"], prediction, labels=[0, 1]).ravel()

    # F1, Precision, Recall과 오분류 건수를 계산하여 기록한다.(42 라인 참고)
    classification_rows.append({"model": name, "scale": "standard", #현재 평가중인 모델의 이름과 스케일링 방식
        # 실패라고 예측한 데이터 중 실제 실패인 비율(정밀도)을 계산
        "F1": f1_score(valid["failed"], prediction, zero_division=0), 
        # 실제 실패 중 모델이 실패라고 정확하게 찾아낸 비율(재현율)을 계산
        "Precision": precision_score(valid["failed"], prediction, zero_division=0),
        # 정밀도와 재현율을 함께 고려한 F1 점수를 계산
        "Recall": recall_score(valid["failed"], prediction, zero_division=0), "FP": fp, "FN": fn})
    
    # 분류 문제의 실제 정답과 예측 결과를 표 형태로 만들어 리스트에 추가(43라인 참고)
    predictions.append(pd.DataFrame({"request_id": valid.index, "task": "classification", "model": name,
        "actual": valid["failed"].to_numpy(), "predicted": prediction}))

#회귀, 분류 평가 결과 리스트를 데이터 프레임으로 변환
regression = pd.DataFrame(regression_rows)
classification = pd.DataFrame(classification_rows)

#폴더 경로 설정 후 생성, 이미 생성되었을 시 그대로 이용
out = HERE / "outputs"
out.mkdir(exist_ok=True)

#회귀, 분류 모델 비교 결과 SCV파일로 저장
regression.to_csv(out / "regression_comparison.csv", index=False, encoding="utf-8-sig")
classification.to_csv(out / "classification_comparison.csv", index=False, encoding="utf-8-sig")

# 각 모델의 예측 결과표를 하나로 합쳐 CSV 파일로 저장
pd.concat(predictions, ignore_index=True).to_csv(out / "validation_predictions.csv", index=False, encoding="utf-8-sig")

#출력
print("회귀 / valid 20건")
print(regression.round(3).to_string(index=False))
print("분류 / valid 20건")
print(classification.round(3).to_string(index=False))
print("저장: outputs의 비교표 2개와 validation_predictions.csv")
print("test는 학습·변환·예측·평가하지 않았습니다.")
