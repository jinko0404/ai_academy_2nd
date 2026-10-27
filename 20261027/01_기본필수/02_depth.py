"""실습 02: 깊이만 바꾸고 같은 train·valid에서 비교합니다."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# 제공: 이 파일 옆의 정상 CSV 20건을 읽습니다. 현재 터미널 위치와 무관합니다.
#파일 읽고 raw에 저장 후 데이터 분할 train/ valid / test = 60/ 20 / 20
HERE = Path(__file__).resolve().parent
raw = pd.read_csv(HERE / "requests.csv")
develop, test = train_test_split(raw, test_size=0.2, stratify=raw["failed"], random_state=42)
train, valid = train_test_split(develop, test_size=0.25, stratify=develop["failed"], random_state=42)
# test 4건은 보관만 합니다. 입력은 실행 전에 아는 글자 수입니다.
X_train = train[["input_chars"]] # 학습 데이터 입력 속성 지정
X_valid = valid[["input_chars"]] # 검증 데이터 입력 속성 지정
y_train = train["failed"] # 학습 데이터의 실제 정답
y_valid = valid["failed"] # 입력 데이터의 실제 정답

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

#깊이 1일때와 3일 때 비교
for depth in [1, 3]:
    # TODO B02: 반복 중인 depth를 max_depth에 연결하세요.
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    #DecisionTreeClassifier는 결정 트리를 이용해 데이터를 분류하는 모델을 만드는 클래스
    # TODO 끝
    model.fit(X_train, y_train)#모델 학습
    # 학습 데이터(X_train)를 예측하고 실제 정답(y_train)과 비교해 F1 점수 계산
    train_f1 = f1_score(y_train, model.predict(X_train), zero_division=0)
    # 검증 데이터(X_valid)를 예측하고 실제 정답(y_valid)과 비교해 F1 점수 계산
    valid_f1 = f1_score(y_valid, model.predict(X_valid), zero_division=0)
    #결과 출력
    print(f"깊이 {depth}: train F1={train_f1:.3f}, valid F1={valid_f1:.3f}")
