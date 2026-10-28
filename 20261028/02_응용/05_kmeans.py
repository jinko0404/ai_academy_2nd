"""실습 05 응용: train 수치2열만 결측 대체·표준화하여 K=3으로 묶습니다."""
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
NUM, CAT = FEATURES[:2], FEATURES[2:]# 앞의 2개는 숫자형 특성, 나머지 2개는 범주형 특성으로 구분
X_train, X_valid = train[FEATURES].copy(), valid[FEATURES].copy()
y_train, y_valid = train["failed"], valid["failed"]

OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 군집화 작업 순서를 하나의 파이프라인으로 구성
cluster = Pipeline([(
    "imputer", SimpleImputer(strategy="median")), # 1단계: 결측값을 해당 열의 중앙값으로 채움
    
    # 2단계: 숫자의 크기 차이를 줄이도록 표준화
    # 예: 입력 글자 수와 허용 도구 수의 단위 차이를 조정
    ("scaler", StandardScaler()),

    # 3단계: 데이터를 3개의 군집으로 나눔
    # n_init=10은 초기 중심점을 바꿔 10번 실행하고 좋은 결과를 선택
    # random_state=42는 실행할 때마다 결과가 달라지는 것을 줄임
    ("kmeans", KMeans(n_clusters=3, n_init=10, random_state=42)
    )])

# TODO A05: 수치2열만으로 cluster를 학습하세요. failed나 elapsed_ms를 넣지 않습니다.
# X_train[NUM]은 input_chars와 allowed_tools 열만 선택함
# fit()은 파이프라인에 지정한 순서대로
# 결측값 대체 → 표준화 → KMeans 군집화를 실행함
# failed나 elapsed_ms는 군집화 입력에 포함하지 않음
cluster.fit(X_train[NUM])
# TODO 끝

# 학습 데이터의 각 행이 어느 군집에 속하는지 가져옴
# 예: [0, 2, 1, 0, ...]처럼 각 데이터의 군집 번호가 저장됨
labels = cluster["kmeans"].labels_

# 학습 데이터의 숫자형 특성 2개를 전처리함
# cluster[:-1]은 마지막 KMeans 단계를 제외하고
# 결측값 대체와 표준화 단계만 실행한다는 뜻
z = cluster[:-1].transform(X_train[NUM])

# KMeans가 계산한 군집 중심점을 원래 숫자 단위로 되돌림
# 표준화된 상태의 중심점을 input_chars와 allowed_tools의 원래 단위로 변환
centers = cluster["scaler"].inverse_transform(cluster["kmeans"].cluster_centers_)

profile = pd.DataFrame(centers, columns=NUM)
profile.insert(0, "cluster", range(3))
profile["n"] = np.bincount(labels, minlength=3)
profile.to_csv(OUT / "cluster_profile.csv", index=False, encoding="utf-8-sig")
pd.DataFrame({"request_id": train_ids, "cluster": labels}).to_csv(OUT / "cluster_assignments.csv", index=False, encoding="utf-8-sig")
print(profile.round(3).to_string(index=False))
print("silhouette(표준화 공간):", round(silhouette_score(z, labels), 3))
print("군집 번호에는 실패·성공·우열의 뜻이 없습니다.")
