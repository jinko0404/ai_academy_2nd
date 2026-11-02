"""
실습 04 · ImageFolder를 이용해 이미지 데이터를 불러오고 DataLoader로 전달한다.

[개념 정리]

1. ImageFolder
- 이미지 데이터셋을 폴더 구조를 기준으로 자동 분류하는 기능이다.
- 각 하위 폴더의 이름을 클래스 이름으로 사용한다.
- 예를 들어 cat 폴더와 dog 폴더가 있다면 각각 하나의 클래스로 인식한다.
- 클래스 이름을 기준으로 숫자 라벨을 자동으로 생성한다.
- root는 이미지가 저장된 최상위 폴더 경로다.
- transform은 이미지에 적용할 전처리 과정이며 선택 사항이다.

2. Transform(변환)
- 이미지 데이터를 모델이 사용하기 적합한 형태로 변경하는 작업이다.
- 이미지 크기 변경, Tensor 변환 등의 전처리를 수행한다.
- 이미지 데이터를 불러올 때 적용할 수 있다.

3. Compose(조합)
- 여러 가지 이미지 전처리 과정을 하나로 묶어주는 기능이다.
- 작성한 순서대로 전처리를 실행한다.
- 예: Resize → ToTensor

4. Resize(크기 변경)
- 이미지의 가로와 세로 크기를 변경하는 기능이다.
- Resize((32, 32))는 이미지를 32×32 크기로 변경한다.
- 목표 크기를 지정해야 한다.

5. ToTensor(Tensor 변환)
- 이미지를 PyTorch Tensor 형태로 변환하는 기능이다.
- 이미지 데이터를 딥러닝 모델이 처리할 수 있는 형태로 바꾼다.
- 일반적인 이미지 입력은 [채널, 높이, 너비] 형태로 변환된다.
- 일반적인 8비트 이미지의 픽셀 값은 0~1 범위로 변환된다.

6. DataLoader
- Dataset에서 데이터를 가져와 Batch 단위로 묶어서 전달하는 기능이다.
- batch_size는 한 번에 전달할 데이터 개수를 의미한다.
- shuffle은 데이터 순서를 무작위로 섞을지 결정한다.
- num_workers는 데이터를 불러오는 별도 프로세스의 수를 설정한다.

7. iter()
- 반복 가능한 객체를 반복자(Iterator)로 변환하는 기능이다.
- DataLoader에서 데이터를 순서대로 가져올 수 있도록 한다.

8. next()
- 반복자에서 다음 데이터를 하나 가져오는 기능이다.
- next(iter(loader))는 DataLoader에서 첫 번째 Batch를 가져온다.

9. class_to_idx
- ImageFolder가 자동으로 생성한 클래스 이름과 숫자 라벨의 대응 관계를 확인하는 속성이다.
- 예: {'cat': 0, 'dog': 1}

10. len()
- 객체에 포함된 데이터의 개수 등을 확인하는 기능이다.
- len(train)은 ImageFolder에 등록된 전체 이미지 개수를 반환한다.

11. shape
- Tensor의 각 차원 크기를 확인하는 속성이다.
- 이미지 Batch의 형태는 일반적으로 [Batch, Channel, Height, Width]다.
"""
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

root = Path(__file__).resolve().parent

# Resize(size)는 목표 크기가 필수. ToTensor()에는 필수 인자가 없다.
# 여러 이미지 전처리 과정을 순서대로 연결한다.
# Resize: 이미지를 32×32 크기로 변경한다.
# ToTensor: 이미지를 PyTorch Tensor 형태로 변환한다.
transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

# ImageFolder(root, transform=None): root는 필수, transform은 선택이다.
# BEGIN B04
# TODO B04: data/train을 주어진 transform으로 읽는다.
# ImageFolder를 이용해 학습용 이미지 데이터셋을 생성한다.
# root / "data" / "train": 이미지가 저장된 폴더 경로
# transform=transform: 이미지를 불러올 때 위에서 설정한 전처리를 적용한다.
train = datasets.ImageFolder(
    root / "data" / "train",
    transform=transform
)
# END B04

# Dataset에 저장된 이미지 데이터를 Batch 단위로 묶어 전달할 DataLoader 생성
# train: 데이터를 가져올 Dataset
# batch_size=4: 한 번에 이미지 4장씩 묶어서 전달
# shuffle=False: 데이터 순서를 섞지 않는다.
# num_workers=0: 메인 프로세스에서 직접 데이터를 불러온다.
loader = DataLoader(
    train,
    batch_size=4,
    shuffle=False,
    num_workers=0
)

# DataLoader를 반복자로 변환한 다음 첫 번째 Batch를 가져온다.
# images: 이미지 데이터
# labels: 이미지에 해당하는 정답 라벨
images, labels = next(iter(loader))

#출력
print("학습 이미지:", len(train))
print("클래스:", train.class_to_idx)
print("배치:", tuple(images.shape), tuple(labels.shape))
print("라벨:", labels.tolist())
