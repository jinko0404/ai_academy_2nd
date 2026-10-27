"""실습 01 응용: 잎 안 정답의 평균으로 숫자를 예측합니다."""
from sklearn.tree import DecisionTreeRegressor

X = [[100], [200], [300], [700], [800], [900]]
y = [100, 120, 140, 700, 720, 740]
model = DecisionTreeRegressor(max_depth=1, random_state=42)
# TODO A01: 회귀 트리를 학습하고 세 입력을 예측하세요.
model.fit(X,y)#X, y 학습
user_input = input("") #데이터 입력 받기
input_nums = user_input.split(" ") # 문자열 형태의 입력 데이터 나누기
z = []# 나눈 데이터 정수형으로 변경 후 넣을 리스트

#입력 받은 데이터 정수형(int)으로 변경 후 리스트에 삽입
for item in input_nums:
    z.append(int(item))

#입력 받은 데이터 3개 예측
prediction = model.predict([[z[0]], [z[1]], [z[2]]])
# TODO 끝

#출력
print("분기 기준:", model.tree_.threshold[0])
print(f"{z[0]}·{z[1]}·{z[2]} 예측:", prediction.tolist())
print("왼쪽 평균:", sum(y[:3]) / 3, "오른쪽 평균:", sum(y[3:]) / 3)
#학습 범위 밖 데이터 안내
for item in z:
    if(item >900):
        print(f"{item}은 학습 범위 밖입니다. 트리도 기존 잎 값으로 예측합니다.")