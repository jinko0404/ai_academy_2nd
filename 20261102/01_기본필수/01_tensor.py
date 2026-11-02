"""실습 01 · RGB 이미지의 N,C,H,W를 확인한다. 실제 사진 대신 정상 Tensor 4장."""
import torch

torch.manual_seed(42)
# 해설 B01: rand의 네 크기는 이미지 수, 채널, 높이, 너비이다.
# 필수: 각 차원의 크기. 선택: dtype, device. 기본 float32·CPU를 사용한다.
# BEGIN B01
# TODO B01: RGB 이미지 4장, 높이·너비 32인 Tensor를 만든다.
"""
Tensor: 숫자를 여러 축으로 묶어 표현하는 자료 구조. 이미지와 모델의 입력·출력을 담는다.
이미지 한 장은 보통 [C, H, W], 여러 장을 묶은 Batch는 [N, C, H, W] 형태를 사용한다.
PyTorch의 Conv2d 입력은 기본적으로 NCHW 순서다.
N	Batch	한 번에 처리하는 이미지 수
C	Channel	색상 또는 특징 채널 수. RGB 이미지는 3
H	Height	이미지 높이
W	Width	이미지 너비
"""
images = torch.rand(4, 3, 32, 32)
# END B01
# 출력
print("shape:", tuple(images.shape))
print("dtype:", images.dtype)
print("device:", images.device)
print("첫 이미지:", tuple(images[0].shape))
