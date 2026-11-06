"""
[실습 03 - A03 기본 개념: 라벨 면적·가로세로비와 데이터 분리 검사]

1. manifest와 데이터 점검
   - manifest.csv는 이미지별 경로, 라벨 경로, 분할, 그룹, 크기, 해시 등을 기록한 표
   - 이 코드는 정상 라벨을 읽어 통계를 계산하며 모델을 학습하거나 추론하지 않음
   - itertuples(): 표의 각 행을 순회하며 image.split처럼 열 이름으로 값을 조회
   - image 변수는 이미지 픽셀 배열이 아니라 manifest의 한 행임

2. YOLO 정답 라벨
   - 한 줄은 class_id cx cy w h로 구성되며 한 물체의 정답 박스를 의미
   - cx, cy는 중심 위치, w, h는 정규화된 폭과 높이
   - read_text(encoding="utf-8"): 문자열 읽기, splitlines(): 줄 목록으로 분리
   - split(): 한 줄을 공백으로 분리, map(float, ...): 각 값을 실수로 변환
   - class_id도 먼저 실수로 읽으므로 저장할 때 int(cls)로 정수 변환

3. 정규화 면적
   - w = 픽셀 박스 폭 / 이미지 너비, h = 픽셀 박스 높이 / 이미지 높이
   - w*h는 전체 이미지 면적에서 박스 면적이 차지하는 비율
   - 예: w=0.2, h=0.5이면 면적 비율은 0.1로 전체의 10%임
   - 이 값은 박스 면적 비율이며 물체 윤곽의 실제 면적을 측정한 것은 아님

4. 픽셀 기준 가로세로비
   - 가로세로비 = 픽셀 단위 박스 폭 / 픽셀 단위 박스 높이
   - 픽셀 폭 = w*image.width, 픽셀 높이 = h*image.height
   - 1보다 크면 가로로 길고, 1보다 작으면 세로로 길며, 1이면 폭과 높이가 같음
   - w/h만 계산하면 정규화 과정에서 서로 다른 축 크기로 나눈 효과가 남음
   - 이미지가 160×128이면 픽셀 기준 비율은 (w/h)의 1.25배
   - 이 실습은 정상적인 양의 박스 높이를 전제로 나눗셈을 수행

5. rows와 결과 표
   - rows.append({...}): 박스마다 하나의 딕셔너리를 리스트에 추가
   - split: 분할 이름, class_id: 클래스 번호, area: 면적 비율
   - aspect_ratio: 원본 픽셀 기준 박스 가로세로비
   - DataFrame(rows): 딕셔너리 키를 열 이름으로 사용하는 표 생성
   - 한 행이 이미지 한 장이 아니라 라벨의 박스 하나에 해당

6. 클래스별 박스 개수
   - groupby(["split", "class_id"]): 분할과 클래스 조합으로 행을 묶음
   - size(): 그룹별 행 수, 즉 정답 박스 수를 계산
   - unstack(fill_value=0): 클래스 번호를 열로 펼치고 없는 조합은 0으로 표시
   - 출력 개수가 맞더라도 CSV의 area·aspect_ratio 계산까지 맞다는 뜻은 아님

7. 그룹 분리와 데이터 누수
   - group_id: 같은 원본·촬영 장면 등으로 묶어 관리하는 식별자
   - 같은 그룹이 여러 split에 있으면 유사 정보가 분할 사이에 섞일 가능성이 있음
   - 그룹별 split.nunique(): 서로 다른 분할 이름의 개수 계산
   - 이 값이 1보다 큰 그룹의 수를 합산하여 출력
   - 0이라고 해도 그룹 정의 밖의 모든 유사 이미지 누수까지 없다고 보장하지는 않음

8. SHA-256과 파일 중복
   - sha256은 파일 바이트를 바탕으로 계산한 해시값
   - duplicated(): 같은 값이 앞에 이미 있으면 True로 표시
   - 합계는 첫 항목을 제외한 중복 행 수이며 중복 그룹 수와 다름
   - 이 코드는 manifest의 해시를 사용하며 현재 파일의 해시를 새로 계산하지 않음
   - 파일 바이트가 다른 시각적 유사 이미지까지 검사하는 것은 아님

9. 빈 라벨과 누락 라벨
   - 빈 라벨: 파일은 존재하지만 물체가 없어 내용이 없는 정상 negative 예시
   - 누락 라벨: 대응 라벨 파일 자체가 없는 상태로 빈 라벨과 구분해야 함
   - 빈 파일은 처리할 줄이 없어 rows에 박스 행이 추가되지 않음
   - 따라서 이 결과 표만으로 negative 이미지 수를 알 수는 없음
   - 마지막 print는 안내 문구이며 누락 라벨을 별도로 진단하는 로직은 아님

10. 경로와 저장
    - Path.resolve().parent: 현재 소스 폴더, / 연산자: 하위 경로 연결
    - mkdir(exist_ok=True): 폴더가 없으면 생성
    - to_csv(index=False): 행 번호 없이 outputs/label_audit.csv에 저장
"""

# 파일 및 폴더 경로 관리
from pathlib import Path
# CSV 읽기, 박스별 결과 표 생성, 그룹별 통계 계산
import pandas as pd
# 터미널 위치와 관계없이 현재 소스 폴더를 기준으로 사용
BASE=Path(__file__).resolve().parent
# 결과 저장 경로를 지정하고 폴더가 없으면 생성
out=BASE/"outputs";out.mkdir(exist_ok=True)
# 이미지별 분할·라벨 경로·너비·높이·그룹·해시가 담긴 목록을 읽음
manifest=pd.read_csv(BASE/"data/manifest.csv")
# 박스별 점검 결과를 모을 리스트
rows=[]
# 이미지 목록을 한 행씩 처리. image는 실제 이미지가 아니라 메타데이터 행임
for image in manifest.itertuples():
    # 해당 라벨 파일을 UTF-8로 읽어 한 줄에 한 박스씩 처리
    for line in (BASE/"data"/image.label).read_text(encoding="utf-8").splitlines():
        # 공백으로 나눈 다섯 값을 실수로 변환하여 클래스·중심·크기로 분리
        cls,cx,cy,w,h=map(float,line.split())
        # >>> A03
        # 요구: 정규화 면적 w*h와 원본 pixel 기준 가로세로비를 기록하세요.
        # TODO A03: 위 요구사항의 핵심 코드를 작성하세요.
        # 현재 박스 정보를 딕셔너리 한 행으로 기록
        rows.append({
            # train·valid·test 등 현재 이미지의 분할 이름
            "split": image.split,
            # 라벨에서 읽은 클래스 값을 정수 번호로 저장
            "class_id": int(cls),
            # 전체 이미지 면적 대비 박스 면적 비율. 픽셀 면적이 아님
            "area": w * h,
            # 각 축의 이미지 크기를 곱해 픽셀 폭·높이 복원 후 비율 계산
            # w/h 또는 (w*h)/(h*h)는 이미지의 서로 다른 너비·높이를 반영하지 못함
            "aspect_ratio": (w * image.width) / (h * image.height)
        })
        # <<< A03
# 모든 박스 결과를 표로 변환하고 행 번호 없이 CSV 저장
r=pd.DataFrame(rows);r.to_csv(out/"label_audit.csv",index=False)
# 분할별·클래스별 박스 수를 표로 출력. 면적·비율은 위 CSV에서 확인
print(r.groupby(["split","class_id"]).size().unstack(fill_value=0).to_string())
# 하나의 group_id가 둘 이상의 split에 나타나는 그룹 개수 출력
print("split 간 그룹 중복:", int((manifest.groupby("group_id").split.nunique()>1).sum()))
# manifest에 기록된 해시가 앞선 행과 중복되는 추가 행의 수 출력
print("정확히 같은 파일:", int(manifest.sha256.duplicated().sum()))
# 물체가 없는 정상 이미지의 빈 라벨과 라벨 파일 누락을 구분하도록 안내
print("빈 라벨은 정상 negative 이미지이며 누락 라벨과 다릅니다.")
