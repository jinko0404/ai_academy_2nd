"""실습 06 응용: 04번에서 만든 CV 표만 읽어 선택하고 평가 계획을 기록합니다."""
from pathlib import Path
import pandas as pd
import json
HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"

# 먼저 04_features.py를 완료하세요. valid_F1을 이용해 후보를 재선택하지 않습니다.
table = pd.read_csv(OUT / "feature_comparison.csv")

# TODO A06: CV_F1이 가장 큰 행의 model 이름을 선택하세요. 정확한 동점은 표의 첫 후보를 사용합니다.
# CV_F1 점수가 가장 높은 행의 인덱스를 찾고, 해당 행의 model 값만 가져옵니다.
# idxmax()는 최댓값이 여러 개면 첫 번째 인덱스를 반환합니다.
selected = table.loc[table["CV_F1"].idxmax(), "model"]
# TODO 끝

# 데이터 분할 정보를 담은 CSV 파일을 읽어옵니다.
manifest = pd.read_csv(HERE / "data/split_manifest.csv")

# train, valid, test 각각에 해당하는 request_id를 가져옵니다.
# split 열에서 해당 구분에 맞는 행만 선택하고,
# position 순서로 정렬한 다음 request_id를 리스트로 만듭니다.
ids = {name: manifest.loc[manifest["split"] == name].sort_values("position")["request_id"].tolist()
       for name in ["train", "valid", "test"]}

# 선택한 모델과 학습·평가에 필요한 설정을 딕셔너리에 정리합니다.
plan = {
    "selected": selected,  # 선택된 모델 이름
    "C": 1,                # Logistic Regression의 규제 강도 관련 설정
    "max_iter": 1000,      # 모델 학습 반복 횟수의 최대값
    "features": ["input_chars", "allowed_tools", "task_type", "priority"],  # 사용할 입력 특성
    "derived": selected == "Logistic_FE",  # 선택 모델이 Logistic_FE인지 여부(True/False)
    "threshold": 0.5,      # 분류 기준 임계값
    "rule": "score > threshold",  # 점수가 임계값보다 클 때 양성으로 판정
    "positive_label": 1,   # 양성으로 취급하는 정답 레이블
    "selection": "train60 5-fold mean F1; exact tie: Logistic first",  # 모델 선택 기준 기록
    "split_ids": ids       # train, valid, test 데이터의 ID 목록
}

# plan 딕셔너리를 JSON 문자열로 변환해 파일에 저장합니다.
# ensure_ascii=False는 한글이 유니코드 이스케이프 형태로 변환되는 것을 방지합니다.
# indent=2는 JSON을 읽기 쉽게 들여쓰기합니다.
(OUT / "selection.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

# 실행결과 화면에 출력
print("선택:", selected)
print("설정: C=1 / 임계값 0.5 초과 / 개발80으로 재학습 / test20 최종 확인")
print("저장: outputs/selection.json (최종평가 이후 변경하지 않습니다.)")
