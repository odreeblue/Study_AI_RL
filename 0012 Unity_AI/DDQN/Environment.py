from Agent import Agent
import numpy as np
import torch
from Env_Racing import Env
import time
import math
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class Environment:
        def __init__(self,gamma, max_steps, num_episodes, num_states, num_actions):
                self.gamma = gamma
                self.max_steps = max_steps
                self.num_epsidoes = num_episodes
                self.num_states = num_states # 태스크의 상태 변수 수(x,z)를 받아옴
                self.num_actions = num_actions # 태스크의 행동 가짓수(위, 아래 ,오른쪽, 왼쪽)를 받아옴
                self.agent = Agent(num_states, num_actions, gamma) # 에이전트 역할을 할 객체를 생성
                self.env = Env() # 

        
        def run(self):
                '''실행'''
                self.env.connect()# 게임과 tcp/ip 연결
                agent_x = 4.0
                agent_z = -4.0
                goal_x = -4.0
                goal_z = 4.0
                initial_position_distance = math.sqrt(math.pow(goal_x-agent_x,2)+math.pow(goal_z-agent_z,2))
                #episode_10_list = np.zeros(10) # 최근 10 에피소드 동안 버틴 단계수를 저장함 --> 평균 산출용
                complete_episodes = 0 #현재까지 목표지점에 도착한 에피소드
                #episode_final = False # 마지막 에피소드 여부
                #frames = [] # 애니메이션을 만들기 위해 마지막 에피소드의 프레임을 저장할 배열
        
                for episode in range(self.num_epsidoes): #최대 에피소드 수만큼 반복
                        #observation = self.env.reset() # 환경 초기화
                        #state = observation
                        time.sleep(2)
                        state = np.array([agent_x,agent_z]) # 관측을 변환없이 그대로 상태 s로 사용  
                        state = torch.from_numpy(state).type(torch.FloatTensor) # Numpy변수를 파이토치 텐서로 변환
                        state = torch.unsqueeze(state, 0)# size 2를 size 1*2 로 변환
                        print(state)
                        #data = torch.zeros(21,21)
                        #position_x = torch.abs(torch.round(state[0,0]-torch.tensor(5)*2))
                        #position_y = torch.abs(torch.round(state[0,1]+torch.tensor(5)*2))
                        #data[position_x.long(),position_y.long()] = torch.tensor(1)
                        #state = torch.reshape(data,(-1,441))
                        for step in range(self.max_steps): # 1에피소드에 해당하는 반복문
                                #if episode_final is True : #마지막 에피소드에서는 각 시각의 이미지를 frames에 저장
                                #        frames.append(self.env.render(mode='rgb_array'))
                                action = self.agent.get_action(state, episode) # 다음 행동을 결정

                                # 행동 a_t를 실행해 다음 상태 s_t+1 과 done 플래그 값을 결정
                                # action에 .item()을 호출해 행동 내용을 구함
                                #observation_next,_,done,_ = self.env.step(action.item()) #reward와 info는 사용하지 않음 _처리
                                if step!=self.max_steps-1: # max_steps에 다다르지 못했을 때
                                        observation_next = self.env.step(action.item(),0)
                                        if observation_next[2]==0: # 충돌없을 때
                                                reward = torch.FloatTensor([0.0])
                                                state_next = np.array([observation_next[0],observation_next[1]])
                                                state_next = torch.from_numpy(state_next).type(torch.FloatTensor)
                                                state_next = torch.unsqueeze(state_next,0)
                                                self.agent.memorize(state,action,state_next, reward)
                                                self.agent.update_q_function()
                                                state = state_next
                                        elif observation_next[2]==1: # 충돌있을 때 
                                                reward = torch.FloatTensor([-1.0])
                                                #state_next = np.array([observation_next[0],observation_next[1]])
                                                #state_next = torch.from_numpy(state_next).type(torch.FloatTensor)
                                                #state_next = torch.unsqueeze(state_next,0)
                                                state_next = None # 충돌하면 다음 행동은 없다
                                                self.agent.memorize(state,action,state_next, reward)
                                                self.agent.update_q_function()
                                                state = state_next
                                                break
                                        elif observation_next[2]==10: # 보너스를 먹으면
                                                reward = torch.FloatTensor([10.0])
                                                state_next = np.array([observation_next[0],observation_next[1]])
                                                state_next = torch.from_numpy(state_next).type(torch.FloatTensor)
                                                state_next = torch.unsqueeze(state_next,0)
                                                #state_next = None # 목표지점에 도착했으니 다음 행동은 없다
                                                self.agent.memorize(state,action,state_next, reward)
                                                self.agent.update_q_function()
                                                #complete_episodes +=1
                                                #print('%d Episode, %d steps에 목표 지점 도착 ! 현재까지 목표지점 도착한 횟수: %d' % (episode,step, complete_episodes))
                                                #break
                                        elif observation_next[2]==15: # 보너스를 먹으면
                                                reward = torch.FloatTensor([20.0])
                                                #state_next = np.array([observation_next[0],observation_next[1]])
                                                #state_next = torch.from_numpy(state_next).type(torch.FloatTensor)
                                                #state_next = torch.unsqueeze(state_next,0)
                                                state_next = None # 목표지점에 도착했으니 다음 행동은 없다
                                                self.agent.memorize(state,action,state_next, reward)
                                                self.agent.update_q_function()
                                                complete_episodes +=1
                                                print('%d Episode, %d steps에 목표 지점 도착 ! 현재까지 목표지점 도착한 횟수: %d' % (episode,step, complete_episodes))
                                                break

                                elif step == self.max_steps-1: # max_steps에 다다랐을때
                                        observation_next = self.env.step(action.item(),1)
                                        #last_position_distance = math.sqrt(math.pow(goal_x-state[0,0].item(),2)+math.pow(goal_z-state[0,1].item(),2))
                                        #reward = (initial_position_distance - last_position_distance)/50
                                        reward = torch.FloatTensor([-0.1])
                                        state_next = None
                                        self.agent.memorize(state,action,state_next,reward)
                                        self.agent.update_q_function()
                                        print('%d Episode, %d steps에도 목표지점 도착 못함 ! 현재까지 목표지점 도착한 횟수: %d' % (episode,step, complete_episodes))
                                        break
                                print('%d Episode, %d steps 진행중! 현재까지 목표지점 도착한 횟수: %d' % (episode,step, complete_episodes))
                        if (episode % 2 ==0):
                                self.agent.update_target_q_function()