# CS492C Deep Learning 2018 SPRING
CS492C 딥러닝 2018년 봄학기 과목 과제 관련 파일입니다.

# Project
MNIST 해결

# Requirements
Tensorflow
Numpy

# Execute
conda activate base
python ~~~.py

# Objective
일반적인 MNIST와는 달리 변형된 숫자 그림 버전의 고급 MNIST를 해결하시오.

# Part
## Part 1
Fully-connected Network Architecture를 이용하여 MNIST를 해결하시오.

### Methods
일반적인 Fully-connected Network를 사용할 경우 Acc가 0.4 전후로 나타난다.
이는 MNIST 데이터가 쉬운 데이터가 아니라 숫자가 뒤집혀있거나 돌려져있는 형태의 데이터가 있는 고난이도의 데이터이기 때문에 그렇다.

따라서 다음과 같은 방법들을 고려해볼 수 있다.
- dropout rate 설정
- L2-regularization scale 설정
- Unit 설정.

하지만 이러한 설정에도 불구하고 Acc가 0.5를 넘지 못한다.
따라서 데이터 상에 문제가 있다고 판단하고 트레이닝 데이터를 다음과 같이 변환하였다.

- Rotate 설정
- Conversion 설정

하지만 normal distribution에 따라 데이터를 augmentation 시킬 경우 Acc가 잘 안 나오는 결과가 있었다. 왜 그런지는 나도 모르겠다. (?)

아무튼 랜덤 ratio를 각도에 따라 조금 변형하였더니.

Acc가 0.672로 상승하였다.
결과는 100여명이 듣는 수업에서 Top 1~3 점수를 받았다.

## Part 2
Convolutional Neural Network Architecture를 이용하여 MNIST를 해결하시오.
또한 3-channel MNIST를 해결하시오.

### Methods
Part 1에서 사용하였던 Data Augmentation를 이용하여 CNN layer를 바로 적용시켰다.

3개의 Layer인데도 불구하고 Acc는 0.817을 기록하였다. 역시 이미지는 뭐다? CNN으로 조지자 이거다.

Layer의 Filter와 Stride를 조정해갔더니,
7-Layer에서는 Acc가 0.941까지 상승하였다.

Data Augmentation이 이로써 고급 MNIST 작업에서 매우 중요했다는 것을 알 수 있다. 매우 허접하게 Data Augmentation을 했는데도 불구하고 100여명이 듣는 수업에서 Top 2를 기록하였다.

수업 과제에서 꿀 빨아서 시험은 잘 못봤지만 나름 괜찮은 grade를 받았다.