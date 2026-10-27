"""실습 04 응용: 전처리 출력 이름 8개와 중요도를 대응시킵니다."""
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
# 숫자, 문자 데이터마다 전처리 방식 적용
pre = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), FEATURES[:2]),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), FEATURES[2:]),
])
# 트리는 이 실습에서 표준화하지 않습니다. 각 후보·fold마다 새 전처리기를 학습합니다.

from sklearn.ensemble import RandomForestClassifier
#전처리와 랜덤 포레스트 모델을 하나의 파이프라인으로 연결한다.
model = Pipeline([
    ("pre", clone(pre)), #기존 전처리(27 ~ 31 라인) 설정 복사해 사용
    ("model", RandomForestClassifier(
        n_estimators=100, #결정 트리 100개 사용
        max_depth=4, #각 트리 최대 깊이 4
        min_samples_leaf=2, #리프 노드에 최소 2개 샘플 데이터 허용
        max_features="sqrt", # 분기마다 고려할 특성 수를 제곱근 기준으로 설정
        random_state=42, # 실행 결과를 재현할 수 있도록 설정
        n_jobs=1 # CPU 작업자 수를 1로 설정
                                    ))
                                    ])
model.fit(X_train, y_train)# 학습 데이터로 전처리 과정과 랜덤 포레스트 모델을 학습
# TODO A04: 전처리의 출력 이름과 Forest의 중요도를 가져오세요.
# 전처리 단계에 접근해 변환된 특성 이름을 가져옴
names = model.named_steps["pre"].get_feature_names_out()

# 랜덤 포레스트 단계에 접근해 각 특성의 중요도를 가져옴
values = model.named_steps["model"].feature_importances_
# TODO 끝
# 특성 이름과 중요도를 하나의 표로 구성
# importance를 기준으로 내림차순 정렬
# 중요도가 높은 특성이 위쪽에 표시됨
result = pd.DataFrame({"feature": names, "importance": values}).sort_values("importance", ascending=False)

# 중요도 결과를 CSV 파일로 저장 후 출력
result.to_csv(OUT / "04_importance.csv", index=False, encoding="utf-8-sig")
print("변환된 특성 수:", len(names))
print(result.round(4).to_string(index=False))
print("중요도는 원인이나 신규 데이터 성능을 보장하지 않습니다.")
