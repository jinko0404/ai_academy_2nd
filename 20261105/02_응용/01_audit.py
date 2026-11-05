"""
[Day14 기본 개념: 데이터셋 클래스 분포 및 중복 검사]

1. ImageFolder
   - 폴더 구조를 기준으로 이미지와 정답 라벨을 불러오는 클래스
   - dataset.classes: 클래스 이름 목록
   - dataset.targets: 각 이미지의 정답 라벨 목록
   - dataset.samples: (이미지 경로, 정답 라벨) 목록

2. defaultdict(list)
   - collections 모듈에서 제공하는 딕셔너리
   - 존재하지 않는 키에 접근하면 빈 리스트를 자동 생성함
   - 동일한 해시값을 가진 이미지 경로를 묶는 데 사용함

3. enumerate()
   - 반복 가능한 객체에서 인덱스와 값을 함께 가져오는 함수
   - enumerate(dataset.classes)는 라벨 번호와 클래스 이름을 반환함

4. count()
   - 리스트에서 특정 값이 등장하는 횟수를 계산하는 메서드
   - dataset.targets.count(label)은 해당 라벨의 이미지 개수를 반환함

5. 데이터셋 클래스 분포
   - 클래스별 이미지 개수와 비율을 확인하는 작업
   - ratio = 해당 클래스 이미지 개수 / 전체 이미지 개수
   - 클래스 불균형 여부를 확인할 때 사용함

6. SHA-256 해시
   - 파일의 바이트 데이터를 고정된 길이의 해시값으로 변환함
   - 동일한 바이트 데이터를 가진 파일은 동일한 해시값을 가짐
   - 해시값을 비교하여 완전히 동일한 파일을 탐지할 수 있음
   - 이미지 내용이 같더라도 압축이나 형식이 다르면 해시값이 달라질 수 있음

7. pandas DataFrame
   - 행과 열로 구성된 2차원 표 형태의 자료구조
   - pd.DataFrame(rows): 딕셔너리 목록을 표로 변환함
   - to_csv(): 데이터를 CSV 파일로 저장함

8. JSON
   - 데이터를 저장하고 교환하는 텍스트 형식
   - json.dumps(): 파이썬 객체를 JSON 문자열로 변환함
   - ensure_ascii=False: 한글을 유니코드 이스케이프 없이 저장함
   - indent=2: 들여쓰기를 적용하여 읽기 쉽게 만듦

9. 데이터 누수(Data Leakage)
   - 학습 과정에 사용하면 안 되는 데이터의 정보가 유입되는 현상
   - train, valid, test 사이에 동일한 이미지가 있는지 확인하는 것이 중요함
   - test 데이터는 모델 학습이나 성능 개선에 사용하지 않아야 함
"""

"""Day14: 제공된 정상 이미지로 학습합니다. test는 읽지 않습니다."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision.datasets import ImageFolder
from torchvision import transforms as tr
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# PyTorch에서 사용하는 CPU 스레드 수를 1개로 설정
torch.set_num_threads(1)

# 난수 시드를 고정하여 실험의 재현성을 높임
torch.manual_seed(14)

# 현재 파이썬 파일이 위치한 폴더 경로
BASE = Path(__file__).resolve().parent

# 데이터셋이 저장된 data 폴더 경로
DATA = BASE / "data"

# 결과 파일을 저장할 outputs 폴더 경로
OUT = BASE / "outputs"

# outputs 폴더가 없으면 생성하고, 이미 존재하면 그대로 사용
OUT.mkdir(exist_ok=True)

# 여기서는 [0,1] RGB 32×32로 학습한 교육용 source 가중치를 사용합니다.
# ImageNet ResNet18과 입력 변환을 섞지 않습니다.

# 이미지 크기를 32×32로 변경한 후 Tensor로 변환하는 전처리 정의
# ToTensor()는 일반적인 8비트 RGB 이미지의 픽셀값을 [0,1] 범위로 변환
transform = tr.Compose([tr.Resize((32, 32)), tr.ToTensor()])

# 기본값을 자동 생성하는 딕셔너리 클래스
from collections import defaultdict

# 파일의 SHA-256 해시값을 계산하기 위한 모듈
import hashlib

# 클래스별 이미지 개수와 비율을 저장할 리스트
rows = []

# 동일한 해시값을 가진 이미지 경로들을 리스트로 묶는 딕셔너리
seen = defaultdict(list)

# train, valid, test 데이터셋을 순서대로 확인
for split in ["train", "valid", "test"]:

    # 현재 데이터셋 폴더에서 이미지 경로와 정답 라벨을 불러옴
    dataset = ImageFolder(DATA / split)

    # 클래스 번호(label)와 클래스 이름(name)을 함께 가져옴
    for label, name in enumerate(dataset.classes):

        # TODO A01: 해당 라벨의 개수를 세세요.
        # BEGIN A01
        count = dataset.targets.count(label)  # dataset.targets에서 label의 수를 세세요.
        # END A01

        # 현재 데이터셋 이름, 클래스 이름, 이미지 개수, 비율을 저장
        # len(dataset)은 현재 데이터셋의 전체 이미지 개수
        rows.append({
            "split": split,
            "class": name,
            "n": count,
            "ratio": count / len(dataset)
        })

    # 데이터셋의 모든 이미지 경로와 라벨을 순회
    # _는 현재 사용하지 않는 정답 라벨을 의미
    for path, _ in dataset.samples:

        # 이미지 파일을 바이트로 읽고 SHA-256 해시값을 계산
        # 동일한 해시값을 가진 이미지 경로를 같은 리스트에 추가
        # relative_to(DATA)는 DATA 기준 상대 경로를 반환
        seen[hashlib.sha256(Path(path).read_bytes()).hexdigest()].append(
            str(Path(path).relative_to(DATA))
        )

# 클래스별 이미지 개수와 비율을 DataFrame으로 변환
table = pd.DataFrame(rows)

# DataFrame을 CSV 파일로 저장
# index=False는 행 번호를 CSV에 저장하지 않도록 설정
table.to_csv(OUT / "class_counts.csv", index=False)

# 비율 등의 숫자를 소수점 넷째 자리까지 반올림하여 출력
# to_string(index=False)는 행 번호 없이 전체 표를 문자열로 변환
print(table.round(4).to_string(index=False))

# 동일한 해시값을 가진 파일이 2개 이상인 그룹만 추출
duplicates = [paths for paths in seen.values() if len(paths) > 1]

# 바이트가 완전히 동일한 이미지 파일 그룹의 개수를 출력
print("바이트 동일 중복 그룹:", len(duplicates))

# 중복 이미지 경로 목록을 JSON 파일로 저장
# ensure_ascii=False는 한글을 그대로 유지
# indent=2는 들여쓰기 2칸 적용
# encoding="utf-8"은 UTF-8 문자 인코딩 사용
(OUT / "duplicates.json").write_text(
    json.dumps(duplicates, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

# test 픽셀은 중복 감사에만 쓰며 모델 성능을 계산하지 않습니다.
