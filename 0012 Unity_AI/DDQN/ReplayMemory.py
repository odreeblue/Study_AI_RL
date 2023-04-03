import random
from collections import namedtuple

class ReplayMemory:
    def __init__(self, CAPACITY): # 생성자
        self.Transition = namedtuple('Transition',('state','action','next_state','reward'))
        self.capacity = CAPACITY # 메모리의 최대 저장 건수 ex. 10000
        self.memory = [] #실제 Transition을 저장할 변수
        self.index = 0 # 저장 위치를 가르칠 인덱스 변수
    def push(self, state, action, state_next, reward):
        '''transition = (state, action, state_next, reward)을 메모리에 저장'''
        if len(self.memory) < self.capacity: # ex ) 100개 < 10000개 일 때
            self.memory.append(None) # 메모리가 가득 차지 않은 경우, memory 마지막에 None을 추가함
                    
        # Transition이라는 namedtuple을 사용해 키-값 쌍의 형태로 값을 저장
        self.memory[self.index] = self.Transition(state, action, state_next, reward)
        # 다음 저장할 위치를 한 자리 뒤로 수정
        self.index = (self.index+1) % self.capacity # %연산자(나머지), 1/10000 = 1, 2/10000 = 2
            
    def sample(self, batch_size):
        ''' batch_size 개수 만큼 무작위로 저장된 transition을 추출'''
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        '''len 함수로 현재 저장된 transition 개수를 반환'''
        return len(self.memory)