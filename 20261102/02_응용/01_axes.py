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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# 이미지 파일 중 첫 번째 PNG 파일의 경로를 가져온다.
# glob("*.png"): 해당 폴더에서 확장자가 .png인 파일을 찾는다.
# sorted(): 파일 경로를 정렬한다.
# [0]: 정렬된 파일 중 첫 번째 파일을 선택한다.
path = sorted((root / "data" / "train" / "horizontal").glob("*.png"))[0]

# 이미지 파일을 열고 RGB 형식으로 변환한 뒤 Tensor로 변경한다.
# Image.open(path): 이미지 파일을 연다.
# convert("RGB"): 이미지를 RGB 3채널 형식으로 변환한다.
# transforms.ToTensor(): 이미지를 PyTorch Tensor로 변환한다.
# 변환된 Tensor의 기본 형태는 CHW(Channel, Height, Width)이다.
chw = transforms.ToTensor()(Image.open(path).convert("RGB"))

# BEGIN A01
# TODO A01: 채널 축을 마지막으로 보내 HWC를 만든다.
#permute(*dims): 기존 축의 순서를 바꾼다. [C,H,W] → [H,W,C]는 permute(1,2,0).
hwc = chw.permute(1,2,0)
# END A01

# HWC 형태의 Tensor를 다시 CHW 형태로 변경한다.
# permute(2, 0, 1): [H,W,C] → [C,H,W]로 축의 순서를 변경한다.
# unsqueeze(0): 맨 앞에 차원을 하나 추가한다.
# 이미지 한 장을 Batch 형태로 만들어 [N,C,H,W] 구조로 만든다.
# 여기서는 이미지가 한 장이므로 N은 1이다.
nchw = hwc.permute(2, 0, 1).unsqueeze(0)

plt.figure(figsize=(4, 4))  # 이미지 출력 영역의 크기를 가로 4인치, 세로 4인치로 설정한다.

# HWC 형태의 이미지 Tensor를 화면에 표시한다.
# imshow()는 이미지 데이터를 시각화하는 함수이다.
# numpy(): PyTorch Tensor를 NumPy 배열로 변환한다.
plt.imshow(hwc.numpy())

plt.title("HWC view: horizontal stripes")   # 그래프 위에 제목을 표시한다.
plt.axis("off")                             # 이미지 주변의 좌표축을 숨긴다.
plt.tight_layout()                          # 이미지가 출력 영역에 알맞게 배치되도록 여백을 조정한다.

# 생성한 이미지를 outputs 폴더에 PNG 파일로 저장한다.
# dpi=120: 이미지 저장 해상도를 설정한다.
plt.savefig(out / "image_axes.png", dpi=120)

plt.close() # 현재 그래프를 닫아 메모리 사용을 정리한다.

# 각 Tensor의 형태를 확인한다.
# tuple(): Tensor의 shape을 읽기 쉬운 튜플 형태로 출력한다.
print("CHW / HWC / NCHW:", tuple(chw.shape), tuple(hwc.shape), tuple(nchw.shape))

# 처음 이미지와 NCHW에서 Batch 차원을 제거한 이미지의 값이
# 완전히 동일한지 확인한다.
# nchw[0]: NCHW에서 첫 번째 이미지 한 장을 꺼낸다.
# torch.equal(): 두 Tensor의 값과 형태가 모두 같은지 검사한다.
# True이면 축을 변경했다가 되돌리는 과정에서 데이터 값이 유지된 것이다.
print("왕복 값 일치:", torch.equal(chw, nchw[0]))
