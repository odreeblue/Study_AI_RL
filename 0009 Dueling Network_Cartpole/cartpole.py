# -*- coding: utf-8 -*-
# Dueling Network
# 기존의 DQN은 어떤 행동을 취하든 받게되는 할인총보상이 상태 s에 의해서만 결정되는 면이있음
# 예를들면, 상태 s가 거의 쓰러지기 직전이라면 행동이 왼쪽이든, 오른쪽이든 봉은 넘어지고, 그에따라 보상의 합계도 매우 적어진다
# 다시 말해, Q함수가 갖는 정보를 상태 s만으로 결정되는 부분과 행동에 따라 결정되는 부분으로 나누어볼수있음
# Dueling Networks는 바로 이 점에 착안해서 Q함수를 상태 s만으로 결정되는 부분 V(s)와 행동에 따라 결정되는 Advantage인 A(s,a)로 나눠서 학습한
# 다음 마지막 출력층에서 V(s)와 A(s,a)를 더해 Q(s,a)를 계산한다.
# DQN과 비교했을 때의 이점은 V(s)로 이어지는 결합 가중치를 행동 a와 상관없이 매 단계마다 학습할 수 있다는 점이다
# 그 덕분에 DQN에 비해 적은 수의 에피소드만으로도 학습을 마칠 수 있다. 이점은 선택 가능한 행동의 가짓수가 늘어날 수록 큰 이점이 된다.


import numpy as np
import matplotlib.pyplot as plt
import gym

# 1. 애니메이션을 만들기
from JSAnimation.IPython_display import display_animation
from matplotlib import animation
from IPython.display import display
def display_frames_as_gif(frames):
    """
    Displays a list of frames as a gif , with controls
    """
    plt.figure(figsize=(frames[0].shape[1]/72.0, frames[0].shape[0]/72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save('movie_cartpole.gif') # The part where i save the animation
    display(display_animation(anim, default_mode = 'loop'))

# 2. 이 코드에서는 namedtuple을 사용함
# named tuple을 사용하면 키-값 쌍 형태로 값을 저장할 수 있음
# 그리고 키를 필드명으로 값에 접근 할 수 있어 편리함
from collections import namedtuple
#Tr = namedtuple('tr', ('name_a', 'value_b'))
#Tr_object = Tr('이름A',100) # 출력 : tr(name_a='이름A', value_b=100)
#print(Tr_object.name_a)  # 출력 : 100
#print(Tr_object.name_a)  # 출력 : 이름A
Transition = namedtuple('Transition',('state','action','next_state','reward'))

# 3. 상수 정의
ENV = 'CartPole-v0' # 태스크 이름
GAMMA  = 0.99 # 시간 할인율
MAX_STEPS =200 #  1에피소드 당 최대 단계 수
NUM_EPISODES = 500 # 최대 에피소드 수

# 4. Transition을 저장하기 위한 메모리 클래스
class ReplayMemory:
        def __init__(self, CAPACITY): # 생성자
                self.capacity = CAPACITY # 메모리의 최대 저장 건수 ex. 10000
                self.memory = [] #실제 Transition을 저장할 변수
                self.index = 0 # 저장 위치를 가르칠 인덱스 변수
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

# 5. 에이전트의 두뇌 역할을 하는 클랫, DQN을 실제 수행함 ###
# Q 함수를 딥러닝 신경망 형태로 정의
import random 
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

BATCH_SIZE = 32
CAPACITY = 10000

class Net(nn.Module):
    def __init__(self,n_in,n_mid,n_out):
        super(Net,self).__init__()# 부모클래스 nn.Module의 __init__을 호출함
        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid,n_mid)
        # Dueling Network
        self.fc3_adv = nn.Linear(n_mid,n_out) # Advantage 함수 쪽 신경망 --> 출력 수는 선택 가능한 행동의 가짓수 n_out과 같음
        self.fc3_v = nn.Linear(n_mid,1) # 가치 V쪽 신경망 --> 상태가치를 나타내므로 출력수는 1임
    
    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))

        adv = self.fc3_adv(h2) # 이출력은 ReLU를 거치지 않음, 출력 크기 32 * 2
        val = self.fc3_v(h2).expand(-1,adv.size(1)) # 이 출력은 ReLU를 거치지 않음, self.fc3_v(h2)의 크기는 32 * 1임 하지만
        # val은 adv와 덧셈을 하기 위해 expand 메서드로 크기를 [minibatch * 1]에서 [minibatch * 2]로 변환
        #                                                       ex. [[1],[2],[3]] ==> [[1,1],[2,2],[3,3]]
        # adv.size(1)은 2(출력할 행동의 가짓수)

        output = val + adv - adv.mean(1, keepdim=True).expand(-1,adv.size(1))
        # val + adv에서 adv의 평균을 뺀다, val의 크기는 32 * 1인데, 늘려서 32 *2 임
                                        #  adv는 출력 그대로 32 * 2임
                                        # 여기서 adv.mean을 빼주는 이유는 adv(s,오른쪽)에 해당하는 바이어스가 b0라 했을 때,
                                        # 이 상태에서 덧셈으로 Q(s,오른쪽)을 제대로 계산하려면 바이어스 b0를 상쇄시키기 위해 V(s)에 바이어스 -b0를 적용해야함
                                        # 다시말해 V(s)와 adv(s,오른쪽)에서 바이어스를 상쇄시킬수 있으므로 어떤 바이어스값이 적용됐더라도 학습이 잘되는 것임
                                        # 한편 왼쪽에 해당하는 바이어스가 b1이라 할때 v(s)부분에 -b1을 적용해야함. 다른말로하면 행동의 종류에 따라 서로 다른 바이어스
                                        # -b0와 -b1이 V(s)에 적용되는 것이다 --> 학습이 불안정해지는 원인임
                                        # 이런 상황을 가능한 한 방지하기 위해 출력값에서 행동의 평균값을 빼는 것임
                                        # 평균값을 빼면 예를 들어 이행동이 오른쪽이라면 다음식과 같이 나타날수있다 ---> 뭔말인지 모르겠음 ........ㅠㅠ
                                        # Q(s,오른쪽) = V(S) + Adv(s,오른쪽) -(Adv(s,오른쪽) + Adv(s, 왼쪽))/2
        # adv.mean(1,keepdim=True)로 열방향(행동의 종류 방향) 평균을 구함, 크기는 [minibatch * 1]이 됨
        # expand 메서드로 크기를 [minibatch * 2]로 늘림
        # ** adv.mean을 출력에서 빼야한다는 것 : 
        return output

# 에이전트의 두뇌 역할을 하는 클래스, DDQN을 실제 수행함
class Brain:
        def __init__(self, num_states, num_actions):
            self.num_actions = num_actions #행동 가짓수(2)를 구함
            
            #transition을 기억하기 위한 메모리 객체 생성
            self.memory = ReplayMemory(CAPACITY)

            #신경망 구성
            n_in, n_mid, n_out = num_states, 32, num_actions
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
            epsilon = 0.5 * (1/ (episode+1))

            if epsilon <= np.random.uniform(0,1):
                self.main_q_network.eval() # 신경망을 추론모드로 전환
                with torch.no_grad():
                    action = self.main_q_network(state).max(1)[1].view(1,1)
                    # 신경망 출력의 최대값에 대한 인덱스 = max(1)[1]
                    # .view(1,1)은 [torch.LongTensor of size 1]을 size 1*1로 변환하는 역할
            else :
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
            expected_state_action_values = self.reward_batch + GAMMA * next_state_values
            
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

class Agent:
        def __init__(self, num_states, num_actions):
                '''태스크의 상태 및 행동의 가짓수를 설정'''
                self.brain = Brain(num_states, num_actions) #Agent's brain role in determining behavior

        def update_q_function(self):
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
        def update_target_q_function(self):
            '''Target Q-Network를 Main Q-Network와 맞춤'''
            self.brain.update_target_q_network()

class Environment:
        def __init__(self):
                self.env = gym.make(ENV) #태스크 설정
                num_states = self.env.observation_space.shape[0] # 태스크의 상태 변수 수(4)를 받아옴
                num_actions = self.env.action_space.n # 태스크의 행동 가짓수(2)를 받아옴
                self.agent = Agent(num_states, num_actions) # 에이전트 역할을 할 객체를 생성
        
        def run(self):
                '''실행'''
                episode_10_list = np.zeros(10) # 최근 10 에피소드 동안 버틴 단계수를 저장함 --> 평균 산출용
                complete_episodes = 0 #현재까지 195단계를 버틴 에피소드 수
                episode_final = False # 마지막 에피소드 여부
                frames = [] # 애니메이션을 만들기 위해 마지막 에피소드의 프레임을 저장할 배열

                for episode in range(NUM_EPISODES): #최대 에피소드 수만큼 반복
                        observation = self.env.reset() # 환경 초기화
                        state = observation # 관측을 변환없이 그대로 상태 s로 사용
                        state = torch.from_numpy(state).type(torch.FloatTensor) # Numpy변수를 파이토치 텐서로 변환
                        state = torch.unsqueeze(state, 0)# size 4를 size 1*4 로 변환

                        for step in range(MAX_STEPS): # 1에피소드에 해당하는 반복문

                                #if episode_final is True : #마지막 에피소드에서는 각 시각의 이미지를 frames에 저장
                                #        frames.append(self.env.render(mode='rgb_array'))
                                action = self.agent.get_action(state, episode) # 다음 행동을 결정

                                # 행동 a_t를 실행해 다음 상태 s_t+1 과 done 플래그 값을 결정
                                # action에 .item()을 호출해 행동 내용을 구함
                                observation_next,_,done,_ = self.env.step(action.item()) #reward와 info는 사용하지 않음 _처리
                                
                                #보상을 부여하고 episode의 종료 판정 및 state_next를 설정
                                if done: # 단계수가 200을 넘었거나 봉이 일정각도 이상 기울면 done이 True가 됨
                                        state_next = None #다음 상태가 없으므로 None 으로 설정

                                        #최근 10episode에서 버틴 단계수를 리스트에 저장
                                        episode_10_list = np.hstack((episode_10_list[1:], step+1))
                                        
                                        if step<195:
                                                reward = torch.FloatTensor([-1.0]) # 도중 봉이 쓰러졌다면 패널티로 -1 부여
                                                complete_episodes = 0 # 연속 성공 에피소드 기록을 초기화
                                        else:
                                                reward = torch.FloatTensor([1.0]) # 봉이 서있는 채로 에피소드 마치면 1 부여
                                                complete_episodes = complete_episodes + 1 # 연속 성공 에피소드 기록을 갱신
                                else :
                                        reward = torch.FloatTensor([0.0]) # 그 외의 경우는 보상 0을 부여
                                        state_next = observation_next # 관측 결과를 그대로 상태로 사용
                                        state_next = torch.from_numpy(state_next).type(torch.FloatTensor) # numpy->torch tensor
                                        state_next = torch.unsqueeze(state_next, 0)#size 4를 size 1*4로 변환
                                
                                # 메모리에 경험을 저장
                                self.agent.memorize(state,action,state_next, reward)
                                # Experience Replay로 Q함수를 수정
                                self.agent.update_q_function()
                                # 관측 결과를 업데이트
                                state = state_next
                                # 에피소드 종료 처리
                                if done:
                                    print('%d Episode: Finished after %d steps : 최근 10 에피소드의 평균 단계 수 = %.1f' % 
                                                                                                    (episode, step+1, episode_10_list.mean()))
                                    
                                    if (episode % 2 ==0):
                                        self.agent.update_target_q_function()
                                    break

                                if episode_final is True:
                                        #애니메이션 생성 및 저장
                                        #display_frames_as_gif(frames)
                                        break
                                # 10 에피소드를 연속으로 195단계를 버티면 태스크 성공
                                if complete_episodes >=10:
                                        print('10에피소드 연속 성공')
                                        episode_final = True # 다음 에피소드에서 애니메이션을 생성

# 실행 엔트리 포인트
cartpole_env = Environment()
cartpole_env.run()




