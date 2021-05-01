# GAN은 학습 시 굉장히 불안정
# (저자: D의 로스가 G이고 G의 로스가 D이기에 불안정??)
# 논문에서 Discriminator의 확률이 올라갔다 내려갔다 하는 등 불안정한 모습을 보이고
# 똑같은 architecture에서 random seed만 다르게 주어서 학습 시 사뮷 다른 plot이 보임
# 이에 Dicriminator를 여러 개 만들어 평균을 하는 방식도 있지만 model이 너무 많아짐(이미 4개)
# 그래서 Reinforcement에서 사용하는 방식 채용 -> 이전의 Generator가 만든 사진들을 주기적으로 Discriminator에게
# 다시 보여주어 D가 예전의 G가 어떻게 행동했는지 까지 대응해야함으로 훨씬 더 안정적인 트레이닝이 됌
# (D의 개수 대신 G의 개수를 늘려주었다고 볼 수 있다)

import random
import time
import datetime
import sys
import numpy as np

from torch.autograd import Variable

import torch
import torchvision.utils

class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful"
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []

        for element in data.data:
            element = torch.unsqueeze(element, 0) # 차원 하나 추가

            if len(self.data) < self.max_size: # dataset[]에 이미지가 아직 다 채워지지않음
                self.data.append(element)
                to_return.append(element)

            else: # dataset[]에 다 채워짐
                if random.uniform(0, 1) > 0.5: # 50퍼 확률
                    i = random.randint(0, self.max_size -1) # 0 ~ 49 사이 랜덤으로 인덱싱
                    to_return.append(self.data[i].clone()) #TODO generated 사진이 중복되지 않나?
                    # dataset list에서 랜덤으로 뽑은 과거 Generated image를 to_return list에 추가
                    self.data[i] = element
                    # 그 자리에 현재 generated image를 대체

                else: # 50퍼 확률
                    to_return.append(element)
                    # 현재 generated image를 to_return에 추가

        return Variable(torch.cat(to_return))
    # Varaible: tensor의 Wrapper (연산그래프에서 Node로 표현)
    # Wrapper Class: 기본 자료형에 대해서 객체로서 인식되도록 '포장' / 객체라는 box에 기본 자료형을 넣은 상태
    # byte, double, float 같은 숫자 자료형의 모든 wrapper 클래스는 Number 추상클래스를 상속 받아서 구현
    # x.dataset:  Tensor의 실제 데이터 접근 / x.grad: x의 변화도를 갖는 또 다른 Variable
    # x.grad_fn: gradient을 계산한 함수에 대한 정보, 어떤 연산에 대한 backward를 진행했는지 저장
    # tensor에 정의딘 거의 모든 연산 지원, .backward() 호출하여 자동으로 모든 기울기 계산
    # tensor와 Varialble => Tensor타입과 병합되어 Tensor타입에서도 디폴트로 autograd 가능
    # 이제는 varaible이 deprecated 상태

class LambdaLR:
    def __init__(self, epochs, offset, decay_start_epoch):
        assert (epochs - decay_start_epoch) > 0, "전체 epoch가 decay_start_epoch보다 커야함"

        self.num_epochs = epochs # 설정한 총 epoch
        self.offset = offset # (저장했었던) start epoch
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch): # epoch : 현재 epoch
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.num_epochs - self.decay_start_epoch)

# 참고 offset: 시작부터 목적지까지 변위 차 e.g) 'c'문자는 A시작점에서 2의 offset 가짐












