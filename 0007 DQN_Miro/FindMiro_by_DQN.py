# -*- coding: utf-8 -*-
# test_02에서 코드를 이쁘게 수정할 것
# 결과는 똑같음
#에이전트의 이동 과정을 시각화
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from collections import namedtuple

from IPython.display import HTML

import random 
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

def init():
    # 배경 이미지 초기화
    line.set_data([],[])
    return (line,)
def animate(i):
    # 프레임 단위로 이미지 생성
    state = s_a_history[i][0] # 현재 위치
    x = (state % 3) + 0.5 # 상태의 x 좌표 : 3으로 나눈 나머지 + 0.5
    y = 2.5 - int(state/3) # y 좌표 : 2.5dptj 3으로 나눈 몫을 뺌
    line.set_data(x,y)
    return (line,)
'''
# 초기화 함수와 프레임 단위 이미지 생성 함수를 사용해 애니메이션 생성
anim = animation.FuncAnimation(fig, animate, init_func = init, frames = len(s_a_history), interval = 200, repeat = False)
HTML(anim.to_jshtml())
'''                  
class Agent:
        def __init__(self, num_states, num_actions):
                '''태스크의 상태 및 행동의 가짓수를 설정'''
                self.brain = Brain(num_states, num_actions) #Agent's brain role in determining behavior **** 완료
                # num_states = 1, num_actions = 4
        def update_Q_function(self):
                '''Modifying the Q function'''
                #self.brain.update_Q_table(observation, action, reward, observation_next)
                self.brain.replay()

        def get_action(self, state, episode):
                '''Action Determination'''
                #action = self.brain.decide_action(observation, step) 
                action = self.brain.decide_action(state, episode) 
                return action                
        def memorize(self, state, action, state_next, reward):
                '''메모리 객체에 state, action, state_next, reward 내용을 저장'''
                self.brain.memory.push(state, action, state_next, reward)
class Brain:
        def __init__(self, num_states, num_actions):
                self.num_actions = num_actions # 행동의 가짓수 (위, 오른쪽,아래, 왼쪽)을 구함 
                
                #transition을 기억하기 위한 메모리 객체 생성
                self.memory = ReplayMemory(CAPACITY)
                
                ###신경망 구성
                self.model = nn.Sequential()

                self.model.add_module('fc1',nn.Linear(num_states, 10)) # 9 X 32 matrix
                self.model.add_module('relu1', nn.ReLU()) # 32 X 32 matrix에 ReLU 함수

                self.model.add_module('fc2',nn.Linear(10, 10)) # 32 X 32 matrix
                self.model.add_module('relu2',nn.ReLU()) # 32 X 32 marix에 ReLU 함수

                self.model.add_module('fc3',nn.Linear(10, num_actions)) # 32 X 4 matrix -> 출력 4개
                print(self.model) # 신경망 구조 출력

                ###최적화 기법 선택
                self.optimizer = optim.Adam(self.model.parameters(), lr = 0.0001)# **** 완료
        
        def replay(self):
                """Experience Replay'로 신경망의 결합 가중치 학습"""
                #----------------------------------------------------
                # 1. 저장된 transition 수 확인
                #----------------------------------------------------
                # 1.1 저장된 transition의 수가 미니배치 크기보다 작으면 아무것도 하지 않음
                if len(self.memory) < BATCH_SIZE:
                        return

                #----------------------------------------------------
                # 2 미니 배치 생성
                #----------------------------------------------------
                # 2.1 메모리 객체에서 미니배치를 추출
                transitions = self.memory.sample(BATCH_SIZE)

                # 2.2 메모리 객체에서 미니배치를 추출
                # transitions는 각 단계 별로 (state, action, state_next, reward) 형태로 BATCH_SIZE 개수만큼 저장됨
                # 다시 말해 (state, action, state_next, reward) * BATCH_SIZE 형태가 된다
                # 이를 미니배치로 만들기 위해
                # (state * BATCH_SIZE, action * BATCH_SIZE, state_next * BATCH_SIZE, reward * BATCH_SIZE) 형태로 변환함
                batch = Transition(*zip(*transitions))# Transition = namedtuple 임
                # ex. [[1,2,3,4],[1,2,3,4],[1,2,3,4]] --> Transition(state=(1, 1, 1), action=(2, 2, 2), next_state=(3, 3, 3), reward=(4, 4, 4))

                # 2.3 각 변수의 요소를 미니배치에 맞게 변형하고, 신경망으로 다룰 수 있게 Variable로 만든다
                # state로 예를 들면, [torch.FloatTensor of size 1*4] 형태의 요소가 BATCH_SIZE 개수만큼 있는 형태임
                # 이를 torch.FloatTensor of size BATCH_SIZE * 4 형태로 변형한다
                # 상태, 행동, 보상, non_final 상태로 된 미니배치를 나타내는 Variable을 생성
                # cat은 Concatenates(연접)을 의미함
                
                state_batch = torch.cat(batch.state)
                action_batch = torch.cat(batch.action)
                #print("action_batch  :",action_batch)
                reward_batch = torch.cat(batch.reward)
                non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])#batch.next_state를 하나씩 s로 꺼내서
                                                                                                 #s가 None이 아닌거만 List로 모아라
                
                #----------------------------------------------------
                # 3 정답신호로 사용할 Q(s_t, a_t)를 계산  
                #----------------------------------------------------
                # 3.1 신경망을 추론 모드로 전환
                self.model.eval()# 신경망 업데이트 안됨

                # 3.2 신경망으로 Q(s_t,a_t)를 계산
                # self.model(state_batch)은 왼쪽, 오른쪽에 대한 Q값을 출력하며 [torch.FloatTensor of size BATCH_SIZE x 2] 형태
                # 여기서부터는 실행한 행동 a_t에 대한 Q값을 계산하므로 action_batch에서 취한 행동 a_t가 왼쪽이냐, 오른쪽이냐에
                # 대한 인덱스를 구하고, 이에 대한 Q값을 gater 메서드로 모아온다
                #print("self.model(state_batch): ", self.model(state_batch))
                state_action_values = self.model(state_batch).gather(1,action_batch)
                #print("self.model(state_batch).gather(1,action_batch): ",self.model(state_batch).gather(1,action_batch))

                # 3.3 max{Q(s_t+1, a)} 값을 계산한다. 이 때 다음 상태가 존재하는지 주의해야 한다.

                # cartpole이 done 상태가 아니고, next_state가 존재하는지 확인하는 인덱스마스크를 만듬
                non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
                #print("non_final_mask  :", non_final_mask)

                # 먼저 전체를 0으로 초기화
                next_state_values = torch.zeros(BATCH_SIZE)

                # 다음상태가 있는 인덱스에 대한 최대 Q값을 구한다
                # 출력에 접근해서 열방향 최댓값(max(1))이 되는 [값, 인덱스]를 구한다
                # 이 Q값(인덱스=0)을 출력한 다음 detach 메서드로 이 값을 꺼내온다
                next_state_values[non_final_mask] = self.model(non_final_next_states).max(1)[0].detach()

                # 3.4 정답 신호로 사용할 Q(s_t, a_t)값을 Q러닝 식으로 계산한다
                expected_state_action_values = reward_batch + GAMMA * next_state_values

                #----------------------------------------------------
                # 4 결합 가중치 수정
                #----------------------------------------------------
                # 4.1 신경망을 학습 모드로 전환
                self.model.train()

                # 4.2 손실함수를 계산(smooth_l1_loss는 Huber 함수)
                # expected_state_action_values는 size가 [minibatch]이므로 unsqueeze해서 [minibatch*1]로 만든다
                loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

                # 4.3 결합 가중치를 수정한다
                self.optimizer.zero_grad() # 경사를 초기화
                loss.backward() #역전파 계산
                self.optimizer.step() # 결합 가중치 수정
                
                #for param in self.model.parameters():
                #      print(param.data)
        def decide_action(self, state, episode):
                '''현재 상태에 따라 행동을 결정한다'''
                # e-greedy 알고리즘에서 서서히 최적행동의 비중을 늘린다
                epsilon = 0.5 * (1/ (episode+1))

                if epsilon <= np.random.uniform(0,1):
                        self.model.eval() # 신경망을 추론모드로 전환
                        with torch.no_grad():
                                action = self.model(state).max(1)[1].view(1,1)
                        # 신경망 출력의 최대값에 대한 인덱스 = max(1)[1]
                        # .view(1,1)은 [torch.LongTensor of size 1]을 size 1*1로 변환하는 역할
                else :
                        # 행동을 무작위로 반환(0 혹은 1)
                        action = torch.LongTensor([[random.randrange(self.num_actions)]])#행동을 무작위로 반환(0 OR 1)
                        # action은 [torch.LongTensor of size 1*1]형태가 된다
                return action
# 4. Transition을 저장하기 위한 메모리 클래스
class ReplayMemory:
        def __init__(self, CAPACITY): # 생성자
                self.capacity = CAPACITY # 메모리의 최대 저장 건수 ex. 10000
                self.memory = [] #실제 Transition을 저장할 변수, Transition : St, At, St+1, Rt+1
                self.index = 0 # 저장 위치를 가르칠 인덱스 변수 ****완료
        def push(self, state, action, state_next, reward):
                '''transition = (state, action, state_next, reward)을 메모리에 저장'''
                if len(self.memory) < self.capacity: # ex ) 100개 < 10000개 일 때
                        self.memory.append(None) # 메모리가 가득 차지 않은 경우, memory 마지막에 None을 추가함
                        
                # Transition이라는 namedtuple을 사용해 키-값 쌍의 형태로 값을 저장
                self.memory[self.index] = Transition(state, action, state_next, reward)

                # 다음 저장할 위치를 한 자리 뒤로 수정
                self.index = (self.index+1) % self.capacity # %연산자(나머지), 1/10000 = 1, 2/10000 = 2
                
        def sample(self, batch_size):
                ''' batch_size 개수 만큼 무작위로 저장된 transition을 추출'''
                return random.sample(self.memory, batch_size)
        
        def __len__(self):
                '''len 함수로 현재 저장된 transition 개수를 반환'''
                return len(self.memory)
GAMMA =0.9 # 시간 할인율
MAX_STEPS = 200 # 1 에피소드 당 최대 단계 수
NUM_EPISODES = 10000 # 최대 에피소드 수
CAPACITY = 10000 # 메모리
Transition = namedtuple('Transition',('state','action','next_state','reward'))
BATCH_SIZE = 32
PATH = "FindMiro.pt"
class Environment:
        def __init__(self):
                #self.env = gym.make(ENV) #태스크 설정
                #num_states = self.env.observation_space.shape[0] # 태스크의 상태 변수 수(4)를 받아옴
                num_states = 9  # 상태 변수 개수는 1개임 --> 1,2,3,4,5,6,7,8
                #num_actions = self.env.action_space.n # 태스크의 행동 가짓수(2)를 받아옴
                num_actions = 4 # 행동 변수 개수는 위, 오른쪽, 아래, 왼쪽 4가지임 
                self.agent = Agent(num_states, num_actions) # 에이전트 역할을 할 객체를 생성 ****완료
        def step(self, state, action):# state , action
                
                state_orig = np.array([ [1,0,0,0,0,0,0,0,0], #s0 -> s1, s3
                                        [0,1,0,0,0,0,0,0,0], #s1 -> s0, s2
                                        [0,0,1,0,0,0,0,0,0], #s2 -> s1    
                                        [0,0,0,1,0,0,0,0,0], #s3 -> s0, s4, s6
                                        [0,0,0,0,1,0,0,0,0], #s4 -> s3, s5, s6
                                        [0,0,0,0,0,1,0,0,0], #s5 -> s4
                                        [0,0,0,0,0,0,1,0,0], #s6 -> s3
                                        [0,0,0,0,0,0,0,1,0], #s7 -> s4, s8
                                        [0,0,0,0,0,0,0,0,1]])#에러상태
                
                cur_state = torch.max(state,1)[1]
                #print(cur_state)
                theta = np.array([[np.nan,1,     1,     np.nan], #s0
                                [np.nan,1,     np.nan,1],      #s1
                                [np.nan,np.nan,1,     1],      #s2     
                                [1,     1,     1,     np.nan], #s3
                                [np.nan,np.nan,1,     1], #s4
                                [1,     np.nan,np.nan,np.nan], #s5
                                [1,     np.nan,np.nan,np.nan], #s6
                                [1,     1,     np.nan,np.nan], #s7
                                [np.nan,np.nan,np.nan,np.nan]]) #s8은 목표지점이므로 정책이 없다.
                next_state = 0
                done = False

                if theta[cur_state,action] == 1:
                    #print("다음액션있다")
                    done = False
                    if action == 0:
                        next_state = state_orig[cur_state-3,:]
                    elif action == 1:
                        next_state = state_orig[cur_state+1,:]
                    elif action == 2:
                        next_state = state_orig[cur_state+3,:]
                    elif action == 3:
                        next_state = state_orig[cur_state-1,:]
                    #print(next_state)
                else:
                    #print("다음액션없다")
                    done = True
                    if cur_state ==8:
                        next_state = [8]
                    else:
                        next_state =[99]
                #print("next_state:", next_state)
                #print("done", done)
                return next_state, done
        
        def run(self):
                '''실행'''
                #episode_10_list = np.zeros(10) # 최근 10 에피소드 동안 버틴 단계수를 저장함 --> 평균 산출용
                #complete_episodes = 0 #현재까지 195단계를 버틴 에피소드 수
                #episode_final = False # 마지막 에피소드 여부
                #frames = [] # 애니메이션을 만들기 위해 마지막 에피소드의 프레임을 저장할 배열
                success = 0
                
                for episode in range(NUM_EPISODES): #최대 에피소드 수만큼 반복
                        #print(str(episode)+ ' Episode Training ....')
                        #observation = self.env.reset() # 환경 초기화
                        #state = observation # 관측을 변환없이 그대로 상태 s로 사용
                        state = [1,0,0,0,0,0,0,0,0] # 초기 위치
                        state = np.array(state) # 추가한 부분
                        state = torch.from_numpy(state).type(torch.FloatTensor) # Numpy변수를 파이토치 텐서로 변환
                        state = torch.unsqueeze(state, 0)# size 9를 size 1*9 로 변환
                        success_state = [0]
                        if success > 1000:
                                torch.save(self.agent.brain.model,PATH)
                                break
                        for step in range(MAX_STEPS): # 1에피소드에 해당하는 반복문
                                
                                print(str(episode)+'Episode '+str(step)+'step training..'+str(success)+"성공!")
                                #if episode_final is True : #마지막 에피소드에서는 각 시각의 이미지를 frames에 저장
                                #        frames.append(self.env.render(mode='rgb_array'))
                                action = self.agent.get_action(state, episode) # 다음 행동을 결정 
                                                                               # 0,1,2,3을 return 받음, torch 1x1 크기

                                # 행동 a_t를 실행해 다음 상태 s_t+1 과 done 플래그 값을 결정
                                # action에 .item()을 호출해 행동 내용을 구함
                                observation_next, done = self.step(state, action.item()) #
                                
                                if done: # 도착지에 도착했거나, 길을 잘못 찾았을 경우 done이 True가됨
                                    state_next = None
                                    if observation_next == [8]:#공이 제대로 도착했다면
                                        reward = torch.FloatTensor([1.0]) # 보상주기
                                        print(str(episode)+"Episode,"+str(step)+" Steps에 목적지 도착!! 현재 위치는 "+str(torch.max(state,1)[1]+8))
                                        success += 1
                                    else:                      #공이 길을 잃었다면
                                        reward = torch.FloatTensor([-1.0])# 벌주기
                                        
                                else:   
                                    reward = torch.FloatTensor([0.0]) # 그외의 경우는 0 부여
                                    state_next = observation_next
                                    #print(state_next)
                                    state_next = torch.from_numpy(state_next).type(torch.FloatTensor)
                                    state_next = torch.unsqueeze(state_next, 0)
                                
                                # 메모리에 경험을 저장
                                self.agent.memorize(state,action,state_next, reward) #*******
                                # Experience Replay로 Q함수를 수정
                                self.agent.update_Q_function()
                                # 관측 결과를 업데이트
                                state = state_next
                                #print(state)
                                if done:
                                    break
                                        
                                #else:
                                #        print('%d Episode \t %d steps \t 1칸 이동 \t state: %.1f, action: %.1f, state_next: %.1f,  Reward : %.1f' 
                                #                      % (episode, step+1,state.item(),action.item(),state_next.item(),reward))
                                ## 10 에피소드를 연속으로 195단계를 버티면 태스크 성공
                                #if complete_episodes >=10:
                                #        print('10에피소드 연속 성공')
                                #        episode_final = True # 다음 에피소드에서 애니메이션을 생성
# 실행 엔트리 포인트
cartpole_env = Environment()
cartpole_env.run()