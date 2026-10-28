"""실습 07~08 응용: 06번 계획을 읽어 개발80으로 새 Pipeline을 학습하고 test20을 평가합니다."""
from pathlib import Path
import json
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
# 06번에서 저장한 모델 선택 계획을 JSON 파일에서 읽음
# 어떤 모델을 선택했는지, 어떤 특성을 사용할지 등의 정보가 들어 있음
plan = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))
raw = pd.read_csv(HERE / "data/requests.csv").set_index("request_id")
ids = plan["split_ids"]
develop = raw.loc[ids["train"] + ids["valid"]]
test = raw.loc[ids["test"]]
X_develop, X_test = develop[plan["features"]].copy(), test[plan["features"]].copy()
NUM, CAT = plan["features"][:2], plan["features"][2:]
# 제공: 같은 행별 공식을 새 입력에도 적용합니다. 학습 통계는 아래 fit(develop)에서만 계산됩니다.

# 06번에서 선택한 모델이 Logistic_FE라면 추가 특성을 생성
if plan["derived"]:

    # 개발 데이터와 테스트 데이터에 같은 계산식을 적용
    # 테스트 데이터의 값으로 전처리 통계를 학습하는 것은 아님
    for part in (X_develop, X_test):

        # input_chars에 로그 변환을 적용한 새로운 특성
        # 큰 값의 차이를 완화하는 데 도움이 될 수 있음
        part["log_input_chars"] = np.log1p(
            part["input_chars"]
        )

        # 허용 도구 수를 고려한 입력 글자 수 관련 특성
        part["chars_per_slot"] = (
            part["input_chars"] / (1.0 + part["allowed_tools"])
        )

    # 새로 생성한 두 특성도 숫자형 전처리 대상에 추가
    NUM = NUM + ["log_input_chars", "chars_per_slot"]

# 전처리 과정 구성
pre = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("scaler", StandardScaler())]), NUM),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                      ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CAT),
])

# 전처리와 모델 학습을 하나로 연결한 Pipeline 생성
model = Pipeline([("pre", pre), ("model", LogisticRegression(C=plan["C"], max_iter=plan["max_iter"]))])

# TODO A07a: 새 Pipeline을 X_develop·failed로 학습하세요. 기존 train 통계를 재사용하지 않고 개발80에서 새로 fit합니다.
# 개발 데이터의 입력 특성과 실제 정답을 이용해 Pipeline 학습
# Pipeline이 먼저 전처리를 학습하고, 변환된 데이터로 분류 모델을 학습함
model.fit(X_develop, develop["failed"])
# TODO 끝

# 실패를 의미하는 정답 레이블 1이
# 모델의 확률 결과에서 몇 번째 열에 해당하는지 확인
positive_column = model.classes_.tolist().index(1)

# TODO A07b: 확정 모델로 test의 실패 확률을 구하세요. test는 fit하지 않습니다.
# 이미 학습한 모델을 사용하므로 여기서는 fit()을 호출하지 않음
# predict_proba()는 각 클래스의 확률을 반환
# [:, 1]은 두 번째 열, 즉 실패 클래스 1의 확률을 선택
probability = model.predict_proba(X_test)[:, 1]
# TODO 끝

# 실패 확률이 임계값보다 크면 1(실패), 그렇지 않으면 0으로 예측
# 예를 들어 threshold가 0.5이고 확률이 0.7이면 실패로 판정
prediction = (probability > plan["threshold"]).astype(int)

# 실제 정답과 예측 결과를 비교해 혼동 행렬 계산
# labels=[0, 1] 순서로 결과를 고정
# ravel()로 네 가지 결과를 각각의 변수에 저장
tn, fp, fn, tp = confusion_matrix(test["failed"], prediction, labels=[0, 1]).ravel()

# 테스트 데이터의 최종 성능 지표를 딕셔너리로 정리
metrics = {
    "n": len(test),  # 테스트 데이터 개수

    # 전체 데이터 중 정답을 맞힌 비율
    "Accuracy": accuracy_score(
        test["failed"], prediction
    ),

    # 실패라고 예측한 것 중 실제 실패인 비율
    "Precision": precision_score(
        test["failed"], prediction, zero_division=0
    ),

    # 실제 실패 중 모델이 찾아낸 비율
    "Recall": recall_score(
        test["failed"], prediction, zero_division=0
    ),

    # Precision과 Recall을 함께 고려하는 지표
    "F1": f1_score(
        test["failed"], prediction, zero_division=0
    ),

    # 실패 확률을 이용해 분류 성능을 평가하는 지표
    "ROC_AUC": roc_auc_score(
        test["failed"], probability
    ),

    # 혼동 행렬의 네 가지 결과
    "TN": int(tn),  # 실제 성공을 성공이라고 예측
    "FP": int(fp),  # 실제 성공을 실패라고 잘못 예측
    "FN": int(fn),  # 실제 실패를 성공이라고 잘못 예측
    "TP": int(tp),  # 실제 실패를 실패라고 예측

    # 테스트 데이터에서 실제로 실패한 건수
    "actual_failed": int(test["failed"].sum())
}


# 테스트 데이터의 개별 예측 결과를 표로 정리
result = pd.DataFrame({
    "request_id": test.index,  # 요청 ID
    "actual": test["failed"].to_numpy(),  # 실제 정답
    "probability_failed": probability,  # 실패 확률
    "predicted": prediction  # 모델의 최종 예측
})

# 개별 예측 결과와 성능 평가 결과를 CSV 파일로 저장
result.to_csv(OUT / "test_predictions.csv", index=False, encoding="utf-8-sig")
(OUT / "test_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

#출력
print("확정 모델:", plan["selected"], "/ 학습80·최종20")
print(pd.DataFrame([metrics]).round(4).to_string(index=False))
print("오류 사례")
print(result.loc[result["actual"] != result["predicted"]].head(3).round(3).to_string(index=False))
print("저장: outputs/test_predictions.csv, test_metrics.json")
print("08번에서는 이 설정·원본을 그대로 재실행합니다. test를 보고 후보를 다시 고르지 않습니다.")
