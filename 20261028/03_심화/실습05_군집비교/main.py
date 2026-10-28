"""실습 05 심화: 같은 표준화 공간에서 K=2·3·4를 비교하고 새 요청에 같은 규칙을 씁니다."""
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
NUM, CAT = FEATURES[:2], FEATURES[2:]# 수치형 특성 2개와 범주형 특성 2개를 구분
X_train, X_valid = train[FEATURES].copy(), valid[FEATURES].copy()
y_train, y_valid = train["failed"], valid["failed"]

OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# K값별 평가 결과를 모아둘 빈 리스트
# K는 군집의 개수를 의미함
rows = []

# 군집을 2개, 3개, 4개로 나눠 각각 비교
for k in [2, 3, 4]:
    # 군집화 파이프라인 구성
    model = Pipeline([
        # 1단계: 결측값을 각 열의 중앙값으로 대체
        ("imputer", SimpleImputer(strategy="median")),

        # 2단계: 특성들의 숫자 크기를 표준화
        ("scaler", StandardScaler()),

        # 3단계: KMeans로 k개 군집 생성
        # n_init=10: 초기 중심점을 바꿔 10번 실행
        # random_state=42: 실행 결과의 재현성을 높임
        ("cluster", KMeans(n_clusters=k, n_init=10, random_state=42))
        ])
    # TODO H05: 현재 K의 model을 train 수치2열로 학습하세요. 비교할 때 전처리와 입력은 고정합니다.
    # X_train[NUM]은 input_chars와 allowed_tools만 선택
    # fit()이 결측값 대체 → 표준화 → 군집화를 순서대로 수행
    # 반복할 때마다 k값이 달라지므로 서로 다른 군집 모델이 만들어짐
    model.fit(X_train[NUM])
    # TODO 끝

    # 마지막 군집화 단계를 제외하고 전처리만 실행
    # z에는 결측값이 대체되고 표준화된 학습 데이터가 들어감
    z = model[:-1].transform(X_train[NUM])

    # 학습 데이터 각각이 배정된 군집 번호 가져오기
    # 예: [0, 1, 1, 0, 2, ...]
    labels = model["cluster"].labels_

    # 현재 K값의 평가 결과를 딕셔너리로 만들어 리스트에 추가
    rows.append({
        "K": k,

        # inertia: 각 데이터와 해당 군집 중심점 사이의
        # 제곱거리 합. 일반적으로 작을수록 군집 내부가 조밀함
        "inertia": model["cluster"].inertia_,

        # silhouette: 군집 내부의 응집도와 군집 간 분리 정도 평가
        # 일반적으로 1에 가까울수록 군집이 잘 분리됨
        "silhouette": silhouette_score(z, labels)
    })

    # K가 3일 때의 모델을 따로 보관
    # 반복문이 끝난 뒤 새 요청을 분류할 때 사용
    if k == 3:
        selected = model
        
result = pd.DataFrame(rows)
result.to_csv(OUT / "cluster_scores.csv", index=False, encoding="utf-8-sig")
centers = selected["scaler"].inverse_transform(selected["cluster"].cluster_centers_)
profile = pd.DataFrame(centers, columns=NUM)
profile.insert(0, "cluster", range(3))
profile["n"] = np.bincount(selected["cluster"].labels_, minlength=3)
profile.to_csv(OUT / "cluster_profile.csv", index=False, encoding="utf-8-sig")
new = pd.DataFrame({"input_chars": [500, 2000], "allowed_tools": [1, 4]})
print(result.round(4).to_string(index=False))
print(profile.round(2).to_string(index=False))
print("새 요청의 K=3 군집:", selected.predict(new).tolist())
print("inertia는 표준화 공간의 제곱거리 합입니다. F1과 평균내지 않습니다.")
