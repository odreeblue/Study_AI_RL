from collections import namedtuple
import torch

#text = "word1anotherword23nextone456lastone333"
#numbers = [x for x in text if x.isdigit()]
#print(numbers)

#t = torch.FloatTensor([[3.0, 1.0,4.0,100.0]])
#print(t.max(1))
#print(t.max(1)[1])
#print(t.max(1)[1].view(1,1))
memory = []
memory.append(None)


Transition = namedtuple('Transition',('state','action','next_state','reward'))
state1 = torch.FloatTensor([[1.1,2.1,3.1,4.1]])
action1 = torch.LongTensor([[1]])
state_next1 = torch.FloatTensor([[5.1,6.1,7.1,8.1]])
reward1 = torch.FloatTensor([0])

state2 = torch.FloatTensor([[1.2,2.2,3.2,4.2]])
action2 = torch.LongTensor([[0]])
state_next2 = torch.FloatTensor([[5.2,6.2,7.2,8.2]])
reward2 = torch.FloatTensor([0])

state3 = torch.FloatTensor([[1.3,2.3,3.3,4.3]])
action3 = torch.LongTensor([[0]])
state_next3 = None
reward3 = torch.FloatTensor([0])

state4 = torch.FloatTensor([[1.4,2.4,3.4,4.4]])
action4 = torch.LongTensor([[0]])
state_next4 = torch.FloatTensor([[5.4,6.4,7.4,8.4]])
reward4 = torch.FloatTensor([0])

memory[0] = Transition(state1, action1, state_next1, reward1)
memory.append(None)
memory[1] = Transition(state2, action2, state_next2, reward2)
memory.append(None)
memory[2] = Transition(state3, action3, state_next3, reward3)
memory.append(None)
memory[3] = Transition(state4, action4, state_next4, reward4)
print("memory : \n", memory)

batch = Transition(*zip(*memory))# state, action, state_next, reward 별로 (1*4) * batch_size 형태로 다시 묶어줌
print("batch : \n", batch)
state_batch = torch.cat(batch.state)# batch_size * 4 형태로 다시 묶어줌
print("state_batch : \n", state_batch)
action_batch = torch.cat(batch.action)
print("action_batch : \n", action_batch)
reward_batch = torch.cat(batch.reward)
print("reward_batch : \n", reward_batch)
non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])#batch.next_state를 하나씩 s로 꺼내서
                                                                                                 #s가 None이 아닌거만 List로 모아라
print("non_final_next_states : \n", non_final_next_states)

import random 
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
num_states = 4
num_actions = 2
class Brain:
    def __init__(self):
        self.model = nn.Sequential()

        self.model.add_module('fc1',nn.Linear(num_states, 32)) # 4 X 32 matrix
        self.model.add_module('relu1', nn.ReLU()) # 4 X 32 matrix에 ReLU 함수

        self.model.add_module('fc2',nn.Linear(32, 32)) # 32 X 32 matrix
        self.model.add_module('relu2',nn.ReLU()) # 32 X 32 marix에 ReLU 함수

        self.model.add_module('fc3',nn.Linear(32, num_actions)) # 32 X 2 matrix -> 출력 2개
                #print(self.model) # 신경망 구조 출력

                ###최적화 기법 선택
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.0001)

brain = Brain()
brain.model.eval()
#state_action_values = brain.model(state_batch).gather(1,action_batch)
print("brain.model(state_batch) : \n",brain.model(state_batch))# Q(St,At)
state_action_values = brain.model(state_batch).gather(1,action_batch)
print("state_action_values : \n",state_action_values) # 상태 행동 가치
print("batch.next_state : \n", batch.next_state)
non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
print("non_final_mask : \n",non_final_mask)

#class A:
#    def __init__(self):
#        self.a = 123
#    def trans(self):
#        self.a = 10
#s = A()
#s.trans()
#print(s.a)
BATCH_SIZE = 4
next_state_values = torch.zeros(BATCH_SIZE)
print("next_state_values : \n",next_state_values)
next_state_values[non_final_mask]= brain.model(non_final_next_states).max(1)[0].detach()
print("brain.model(non_final_next_states):\n",brain.model(non_final_next_states))
print("brain.model(non_final_next_states).max(1):\n",brain.model(non_final_next_states).max(1))
print("brain.model(non_final_next_states).max(1)[0]:\n",brain.model(non_final_next_states).max(1)[0])
print("brain.model(non_final_next_states).max(1)[0].detach():\n",brain.model(non_final_next_states).max(1)[0].detach())
print("next_state_values : \n",next_state_values)

GAMMA = 0.99
expected_state_action_values = reward_batch + GAMMA * next_state_values
print("expected_state_action_value : \n",expected_state_action_values)
