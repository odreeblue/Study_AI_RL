from ReplayMemory import ReplayMemory
from collections import namedtuple
from Net import Net
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import random 
# 에이전트의 두뇌 역할을 하는 클래스, DDQN을 실제 수행함
BATCH_SIZE = 32
CAPACITY = 10000
Transition = namedtuple('Transition',('state','action','next_state','reward'))
class Brain:
        def __init__(self, num_states, num_actions,gamma):
            self.num_actions = num_actions #행동 가짓수(2)를 구함
            self.gamma = gamma
            #transition을 기억하기 위한 메모리 객체 생성
            self.memory = ReplayMemory(CAPACITY)

            #신경망 구성
            n_in, n_mid, n_out = num_states, 200, num_actions
            self.main_q_network = Net(n_in, n_mid, n_out) # Net 클래스 사용
            self.target_q_network = Net(n_in, n_mid, n_out) # Net 클래스 사용
            print(self.main_q_network)
            
            # 최적화 기법 선택
            self.optimizer = optim.Adam(self.main_q_network.parameters(), lr = 0.0001)
        
        def replay(self):
            """Experience Replay'로 신경망의 결합 가중치 학습"""
            #----------------------------------------------------
            # 1. 저장된 transition 수 확인
            #----------------------------------------------------
            if len(self.memory) < BATCH_SIZE:
                return

            #----------------------------------------------------
            # 2 미니 배치 생성
            #----------------------------------------------------
            self.batch, self.state_batch, self.action_batch, self.reward_batch, self.non_final_next_states = self.make_minibatch()
                
            #----------------------------------------------------
            # 3 정답 신호로 사용할 Q(s_t,a_t)를 계산
            #----------------------------------------------------
            self.expected_state_action_values = self.get_expected_state_action_values()

            #----------------------------------------------------
            # 4 결합 가중치 수정
            #----------------------------------------------------
            self.update_main_q_network()
        
        def decide_action(self, state, episode):
            '''현재 상태에 따라 행동을 결정한다'''
            # e-greedy 알고리즘에서 서서히 최적행동의 비중을 늘린다
            epsilon = 0.5 * (1/ (episode/500+1))
            #print(epsilon)
            if epsilon <= np.random.uniform(0,1):
                #print("network")
                self.main_q_network.eval() # 신경망을 추론모드로 전환
                with torch.no_grad():
                    #print("self.main_q_network(state) : ",self.main_q_network(state))
                    #print(".max(1): ",self.main_q_network(state).max(1))
                    #print(".max(1)[1]: ",self.main_q_network(state).max(1)[1])
                    #data = torch.zeros(21,21)
                    #torch.LongTensor([[random.randrange(self.num_actions)]])
                    #position_x = torch.abs(torch.round(state[0,0]-torch.tensor(5)*2))
                    #position_y = torch.abs(torch.round(state[0,1]+torch.tensor(5)*2))
                    #data[position_x.long(),position_y.long()] = torch.tensor(1)
                    #data = torch.reshape(data,(-1,))
                    action = self.main_q_network(state).max(1)[1].view(1,1)
                    # 신경망 출력의 최대값에 대한 인덱스 = max(1)[1]
                    # .view(1,1)은 [torch.LongTensor of size 1]을 size 1*1로 변환하는 역할
            else :
                #print("random")
                # 행동을 무작위로 반환(0 혹은 1)
                action = torch.LongTensor([[random.randrange(self.num_actions)]])#행동을 무작위로 반환(0 OR 1 OR 2 OR 3)
                # action은 [torch.LongTensor of size 1*1]형태가 된다
            return action
        
        def make_minibatch(self):
            '''2. 미니 배치 생성'''
            # 2.1 메모리 객체에서 미니배치를 추출
            transitions = self.memory.sample(BATCH_SIZE)
            
            # 2.2 각 변수를 미니배치에 맞는 형태로 변형
            # transitions는 각 단계별로(state, action, state_next, reward)형태로 BATCH_SIZE 개수만큼 저장됨
            # 다시 말해, (state, action, state_next, reward) * BATCH_SIZE
            # 미니 배치로 만들기 위해
            # (state * BATCH_SIZE, action*BATCH_SIZE, state_next*BATCH_SIZE, reward* BATCH_SIZE) 형태로 변환
            batch = Transition(*zip(*transitions))

            # 2.3 각 변수의 요소를 미니배치에 맞게 변형하고, 신경망으로 다룰수 있게 Variable로 만든다
            # state를 예로 들면, [torch.FloatTensor of size 1*4] 형태의 요소가 BATCH_SIZE 개수만큼 있는 형태이다.
            # 이를 torch.FloatTensor of size BATCH_SIZE * 4 형태로 변형한다.
            # 상태, 행동, 보상, non_final 상태로 된 미니 배치를 나타내는 Variable 을 생성
            # cat은 Concatenates(연접)을 의미함
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

            return batch, state_batch, action_batch, reward_batch, non_final_next_states

        def get_expected_state_action_values(self):
            '''3. 정답 신호로 사용할 Q(s_t, a_t)를 계산'''
            # 3.1 신경망을 추론 모드로 전환
            self.main_q_network.eval()
            self.target_q_network.eval()

            # 3.2 신경망으로 Q(s_t, a_t)를 계산
            # self.model(state_bach)은 왼쪽, 오른쪽에 대한 Q 값을 출력하며
            # [torch.FloatTensor of size BATCH_SIZE * 2] 형태다
            # 여기서부터는 실행한 행동 a_t에 대한 Q값을 계산하므로 action_batch에서 취한 행동 a_t가 왼쪽이냐 오른쪽이냐
            # 에 대한 인덱스를 구하고, 이에 대한 Q 값을 gather메서드로 모아온다
            self.state_action_values = self.main_q_network(self.state_batch).gather(1, self.action_batch)
            #print(self.main_q_network(self.state_batch))
            #print(self.action_batch)
            #print(self.state_action_values)
            #return
            # 3.3 max{Q(s_t+1, a)}값을 계산한다. 이때 다음상태가 존재하는지에 주의해야한다.
            
            # cartpole이 done 상태가 아니고, next_state가 존재하는지 확인하는 인덱스 마스크를 만든다
            non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, self.batch.next_state)))
            #print("self.batch.next_state ",self.batch.next_state)
            #print("non_final_mask ",non_final_mask)
            #return
            # 먼저 전체를 0으로 초기화
            next_state_values = torch.zeros(BATCH_SIZE)
            a_m = torch.zeros(BATCH_SIZE).type(torch.LongTensor)

            # 다음 상태에서 Q값이 최대가 되는 행동 a_m을 Main Q-Network로 계산
            # 마지막에 붙은 [1]로 행동에 해당하는 인덱스를 구함
            a_m[non_final_mask] = self.main_q_network(self.non_final_next_states).detach().max(1)[1]
            #print("self.main_q_network(self.non_final_next_states): ", self.main_q_network(self.non_final_next_states))
            #print("self.main_q_network(self.non_final_next_states).detach().max(1)[1]: ",self.main_q_network(self.non_final_next_states).detach().max(1)[1])
            #print("a_m[non_final_mask]: ",a_m[non_final_mask])
            # 다음 상태가 있는 것만을 걸러내고, size 32를 32*1로 변환
            a_m_non_final_next_states = a_m[non_final_mask].view(-1,1)
            #print("a_m_non_final_next_states: ",a_m_non_final_next_states)
            
            # 다음 상태가 있는 인덱스에 대해 행동 a_m의 Q값을 target Q-Network로 계산
            # detach() 메서드로 값을 꺼내옴
            # squeeze() 메서드로 size[minibatch * 1]을 [minibatch]로 변환
            next_state_values[non_final_mask] = self.target_q_network(self.non_final_next_states).gather(1, a_m_non_final_next_states).detach().squeeze()

            # 3.4 정답 신호로 사용할 Q(s_t, a_t) 값을 Q 러닝 식으로 계산
            expected_state_action_values = self.reward_batch + self.gamma * next_state_values
            
            return expected_state_action_values
        def update_main_q_network(self):
            '''4. 결합 가중치 수정'''
            # 4.1 신경망을 학습 모드로 전환
            self.main_q_network.train()
            # 4.2 손실함수를 계산(smooth_l1_loss 는 huber 함수)
            # expected_state_action_values는 size가 [minibatch]이므로 unsqueeze해서 [minibatch *1]로 만듦
            loss = F.smooth_l1_loss(self.state_action_values, self.expected_state_action_values.unsqueeze(1))
            # 4.3 결합 가중치를 수정
            self.optimizer.zero_grad() #경사를 초기화
            loss.backward() #역전파 계산
            self.optimizer.step() # 결합 가중치 수정
        def update_target_q_network(self):
            '''Target Q-Network을 Main Q-Network와 맞춤'''
            self.target_q_network.load_state_dict(self.main_q_network.state_dict())