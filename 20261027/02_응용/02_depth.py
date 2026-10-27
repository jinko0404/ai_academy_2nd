"""실습 02 응용: 기존 PPT의 5개 깊이 후보를 비교합니다."""
from pathlib import Path
import pandas as pd

# 제공: 전날의 분할 소속뿐 아니라 position 순서까지 그대로 읽습니다.
# 현재 파이썬 파일이 있는 폴더의 경로
HERE = Path(__file__).resolve().parent

# requests.csv를 읽고 request_id를 행의 인덱스로 설정
# 인덱스로 설정하면 request_id를 이용해 원하는 행을 찾을 수 있음
raw = pd.read_csv(HERE / "data/requests.csv").set_index("request_id")

#미리 정해놓은 학습 데이터와 검증데이터 분할 정보 읽기
manifest = pd.read_csv(HERE / "data/split_manifest.csv")

# 각각 split이 train, vaild인 행만 선택하고,
# position 순서대로 정렬한 뒤 해당 request_id만 가져옴
train_ids = manifest.loc[manifest["split"] == "train"].sort_values("position")["request_id"]
valid_ids = manifest.loc[manifest["split"] == "valid"].sort_values("position")["request_id"]

#앞에서 가져온 요청 id값에 따라 원본 데이터에서 학습, 검증 데이터 가져오기
#미리 정해진 행 순서와 데이터 분할 유지
train = raw.loc[train_ids]
valid = raw.loc[valid_ids]

FEATURES = ["input_chars", "allowed_tools", "task_type", "priority"]#모델 학습에서 사용할 입력 특성 4가지 지정
X_train, X_valid = train[FEATURES], valid[FEATURES] # X는 모델에 넣을 입력 데이터
y_train, y_valid = train["failed"], valid["failed"] # y는 모델이 예측해야할 정답, 예측할 정답인 failed열 추출
OUT = HERE / "outputs" # 결과 저장할 폴더 지정
OUT.mkdir(exist_ok=True) # 폴더가 이미 있으면 그대로 이용한다.
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
from sklearn.metrics import f1_score

rows = []#깊이별 실험 결과를 저장할 빈 리스트

for depth in [1, 2, 3, 5, None]:# 결정 트리의 깊이를 1, 2, 3, 5, 제한 없음으로 바꾸어 비교
    # None은 최대 깊이를 별도로 제한하지 않는다는 의미
    # TODO A02: depth를 실제 트리 깊이로 적용하세요.
    estimator = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=1, random_state=42)
    # TODO 끝
    # 전처리 후 결정 트리를 학습하는 파이프라인 생성
    model = Pipeline([("pre", clone(pre)), ("model", estimator)])

    # 학습 데이터로 전처리 과정과 결정 트리 모델을 학습
    model.fit(X_train, y_train)

    # 현재 깊이의 학습 점수, 검증 점수, 리프 노드 수를 저장
    rows.append({"depth": str(depth), "train_F1": f1_score(y_train, model.predict(X_train), zero_division=0),
                 "valid_F1": f1_score(y_valid, model.predict(X_valid), zero_division=0),
                 "leaves": model["model"].get_n_leaves()})
result = pd.DataFrame(rows)
result.to_csv(OUT / "02_depth.csv", index=False, encoding="utf-8-sig")
print(result.round(3).to_string(index=False))
