"""
실습 A04: 이미지 증강(Image Augmentation)

[새로운 개념]

1. Image Augmentation(이미지 증강)
   - 원본 이미지에 여러 변형을 적용해 학습 데이터를 다양하게 만드는 방법
   - 좌우 반전, 회전, 밝기 변경 등이 대표적인 방법

2. transforms.Compose()
   - 여러 이미지 변환 작업을 하나로 묶는 기능
   - 등록된 순서대로 변환 작업을 실행한다.

3. RandomHorizontalFlip()
   - 이미지를 일정 확률로 좌우 반전한다.
   - 실행할 때마다 반전 여부가 달라질 수 있다.

4. RandomRotation(10)
   - 이미지를 -10도부터 +10도 사이의 임의의 각도로 회전한다.

5. ColorJitter(brightness=0.15)
   - 이미지의 밝기를 무작위로 변경한다.

6. ToTensor()
   - 이미지를 PyTorch 텐서 형태로 변환한다.
   - 이미지의 픽셀값은 일반적으로 0~1 범위로 변환된다.

7. ToPILImage()
   - PyTorch 텐서를 다시 PIL 이미지 형태로 변환한다.

8. PIL(Image)
   - 파이썬에서 이미지를 열고, 수정하고, 저장하는 기능을 제공한다.

9. ImageDraw
   - 이미지 위에 글자나 도형을 그리는 기능을 제공한다.

10. torch.equal()
    - 두 텐서의 값이 완전히 같은지 비교한다.
    - 모든 값이 같으면 True, 하나라도 다르면 False를 반환한다.

11. torch.manual_seed(42)
    - 난수 시드를 고정해 무작위 작업의 재현성을 높인다.

12. enumerate()
    - 반복문에서 데이터와 해당 데이터의 인덱스를 함께 가져온다.

13. glob("*.png")
    - 지정한 폴더에서 PNG 파일을 검색한다.

14. Image.new()
    - 새로운 빈 이미지를 생성한다.

15. ImageDraw.Draw()
    - 생성한 이미지 위에 글자를 그릴 수 있도록 한다.

[학습용과 검증용의 차이]
- train_transform: 학습 이미지에 무작위 변형을 적용한다.
- valid_transform: 검증 이미지에는 Resize와 ToTensor만 적용한다.
- 검증 단계에서는 이미지 변형을 최소화해 일관된 평가가 가능하도록 한다.
"""

from pathlib import Path
import copy
import csv
import json
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pandas as pd

torch.manual_seed(42)
torch.set_num_threads(1)
root = Path(__file__).resolve().parent
out = root / "outputs"
out.mkdir(exist_ok=True)

from PIL import Image, ImageDraw

path = sorted((root / "data" / "train" / "horizontal").glob("*.png"))[0]

image = Image.open(path).convert("RGB") # 선택한 이미지 파일을 열고 RGB 색상 형식으로 변환한다.

# 좌우반전은 가로·세로 줄무늬 라벨을 보존한다.
# 90도 회전은 라벨을 바꾸므로 사용하지 않는다.
# 여기서는 작은 각도인 10도 범위의 회전을 적용한다.

# 학습용 이미지에 적용할 변환 작업을 묶는다.
train_transform = transforms.Compose([
    transforms.Resize((64, 64)),              # 이미지 크기를 64x64로 변경한다.
    transforms.RandomHorizontalFlip(),        # 이미지를 무작위로 좌우 반전한다.
    transforms.RandomRotation(10),            # -10도~+10도 사이에서 무작위 회전한다.
    transforms.ColorJitter(brightness=0.15),  # 이미지 밝기를 무작위로 변경한다.
    transforms.ToTensor(),                    # 이미지를 PyTorch 텐서로 변환한다.
])

# BEGIN A04
# TODO A04: valid에는 Resize와 ToTensor만 적용한다.
# 검증용 이미지에는 무작위 증강을 적용하지 않는다.
# 이미지 크기 변경과 텐서 변환만 수행한다.
valid_transform = transforms.Compose([
    transforms.Resize((64, 64)),  # 이미지 크기를 64x64로 변경한다.
    transforms.ToTensor()         # 이미지를 PyTorch 텐서로 변환한다.
])
# END A04

# 원본 이미지에 학습용 변환을 3번 적용한다.
# 실행할 때마다 좌우 반전, 회전, 밝기 변경 결과가 달라질 수 있다.
# 따라서 같은 원본 이미지에서 3개의 증강 이미지를 만든다.
augmented = [train_transform(image) for _ in range(3)]

# 같은 원본 이미지에 검증용 변환을 각각 적용한다.
# 검증용 변환에는 무작위 작업이 없으므로 두 결과는 동일해야 한다.
v1, v2 = valid_transform(image), valid_transform(image)

# 결과 이미지를 나란히 보여주기 위한 빈 캔버스를 생성한다.
# 가로 크기: 128픽셀 × 4개 이미지
# 세로 크기: 150픽셀
# 배경색: 흰색
canvas = Image.new("RGB", (4*128, 150), "white")

draw = ImageDraw.Draw(canvas) # 캔버스 위에 글자를 그리기 위한 객체를 생성한다.

# 검증 이미지 1개와 학습 증강 이미지 3개를 순서대로 확인한다.
# enumerate()는 이미지와 이미지의 순서(인덱스)를 함께 가져온다.
for i, tensor in enumerate([v1]+augmented):

    # 텐서를 PIL 이미지로 변환한다.
    # 미리보기 크기에 맞게 128x128로 확대한다.
    # 캔버스에 이미지를 붙일 위치는 (128*i, 22)이다.
    canvas.paste(
        transforms.ToPILImage()(tensor).resize((128,128)),
        (128*i,22)
    )

    # 첫 번째 이미지는 valid라고 표시한다.
    # 나머지 이미지는 train 1, train 2, train 3으로 표시한다.
    draw.text(
        (128*i+5,4),
        "valid" if i==0 else f"train {i}",
        fill="black"
    )

canvas.save(out / "augmentation_preview.png")   # 완성된 미리보기 이미지를 outputs 폴더에 저장한다.

# 출력
print("valid 동일:", torch.equal(v1, v2))
print("shape:", tuple(v1.shape))
print("train 3개 동일 여부:", torch.equal(augmented[0], augmented[1]))
