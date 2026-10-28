"""실습 06 심화: train CV로 선택하고 설정을 저장합니다. test 입력·정답은 선택하지 않습니다."""
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

import json
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score
from preprocessing import CANDIDATES

# 제공: 최종 평가 파일이 이미 있으면 기존 계획을 보존합니다. 보안 잠금이 아닌 실습용 진행 안내입니다.
if (OUT / "test_metrics.json").exists():
    print("최종 평가 이후에는 선택을 다시 하지 않습니다. 기존 selection.json으로 main.py를 재실행하세요.")
else:
    # train 데이터를 5개로 나눠 교차 검증을 준비
    # shuffle=True: 데이터를 무작위로 섞음
    # random_state=42: 같은 조건에서 동일한 분할을 재현
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_train, y_train))

    # 후보 모델별 평균 점수 등을 저장할 리스트(rows)
    # 각 모델의 회차별 점수를 저장할 리스트(fold_rows)
    rows, fold_rows = [], []

    # CANDIDATES에 등록된 후보 모델을 하나씩 꺼냄
    for name, prototype in CANDIDATES.items():
        # 후보 모델을 복사
        # 원본 설정을 유지하면서 실험에 사용할 모델을 준비
        model = clone(prototype)

        # TODO H06: 각 후보 Pipeline을 동일 train 5-fold로 비교하세요. 미리 전체 train을 변환하지 않습니다.
        # F1 점수 5개를 저장할 공간
        scores = np.zeros(5) # 처음에는 [0, 0, 0, 0, 0]
        
        # folds에 저장된 5개 분할을 하나씩 꺼냄
        # i는 회차 인덱스(0~4)
        # train_idx는 해당 회차의 학습용 행 인덱스
        # valid_idx는 해당 회차의 검증용 행 인덱스
        for i, (train_idx, valid_idx) in enumerate(folds):

            # 이번 회차에서 사용할 모델을 새로 복사
            # 각 회차는 독립적으로 학습해야 하므로 새 모델 사용
            model_fold = clone(prototype)

            # 이번 회차의 학습용 데이터로 모델 학습
            # X는 입력 특성, y는 정답
            model_fold.fit(
                X_train.iloc[train_idx],
                y_train.iloc[train_idx]
            )

            # 이번 회차의 검증용 입력 데이터로 예측
            pred = model_fold.predict(
                X_train.iloc[valid_idx]
            )

            # 예측 결과와 실제 정답을 비교해 F1 점수 계산
            # 계산한 점수를 해당 회차의 위치에 저장
            scores[i] = f1_score(
                y_train.iloc[valid_idx],
                pred,
                zero_division=0
            )

            
        # TODO 끝

        # 5회 점수의 평균과 표준편차를 계산
        # valid_F1은 별도로 valid 데이터에서 측정한 참고용 점수
        # 모델 선택 기준은 CV_F1
        model.fit(X_train, y_train)
        rows.append({
            "model": name,
            "CV_F1": scores.mean(),
            "CV_std": scores.std(),
            "valid_F1": f1_score(
                y_valid,
                model.predict(X_valid),
                zero_division=0
            )
        })

        # 각 회차의 F1 점수를 개별 기록
        for number, value in enumerate(scores, start=1):
            fold_rows.append({
                "model": name,
                "fold": number,
                "F1": value
            })

    result = pd.DataFrame(rows) # 모델별 교차 검증 결과를 표로 변환

    # CV_F1 평균이 가장 높은 모델 선택
    # 동점이면 먼저 등장한 후보가 선택됨
    selected = result.loc[result["CV_F1"].idxmax(), "model"]

    # test 데이터의 ID를 가져옴, 여기서는 ID만 기록하고, test 데이터로 모델을 선택하지 않음
    test_ids = manifest.loc[manifest["split"] == "test"].sort_values("position")["request_id"].tolist()

    # 최종 실행에 필요한 모델 선택 정보와 설정을 딕셔너리에 정리
    plan = {
        "selected": selected,
        "candidate_order": list(CANDIDATES),
        "features": FEATURES,
        "derived": selected == "Logistic_FE",
        "threshold": 0.5,
        "rule": "score > threshold",
        "positive_label": 1,
        "selection": "train60 5-fold mean F1 max; exact tie: candidate order",
        "cv_splits": 5,
        "cv_shuffle": True,
        "cv_random_state": 42,
        "train_ids": train_ids,
        "valid_ids": valid_ids,
        "test_ids": test_ids,
        "model_params": CANDIDATES[selected]["model"].get_params(deep=False)
    }
    # 후보별 평균 점수와 표준편차를 CSV로 저장
    result.to_csv(
        OUT / "cv_comparison.csv",
        index=False,
        encoding="utf-8-sig"
    )
    # 후보별 회차 점수를 CSV로 저장
    pd.DataFrame(fold_rows).to_csv(
        OUT / "cv_scores.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Logistic과 Logistic_FE의 비교 결과만 별도 저장
    result.loc[
        result["model"].isin(["Logistic", "Logistic_FE"])
    ].to_csv(
        OUT / "feature_comparison.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 선택 결과와 설정을 JSON 파일로 저장
    (OUT / "selection.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # 결과 출력
    print(result.round(4).to_string(index=False))
    print("선택:", selected, "/ 계획 저장 완료 / test20 미평가")
