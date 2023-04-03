# -*- coding: utf-8 -*-
# Package import to be used for implementation
from csv import writer
import numpy as np
import matplotlib.pyplot as plt
import gym

# A function that makes animations
from JSAnimation.IPython_display import display_animation
from matplotlib import animation
from IPython.display import display


# Define the Constant 
ENV = 'CartPole-v0' # Task Name
NUM_DIZITIZED =6 # Number of segments to convert each state to a discrete variable
GAMMA= 0.99 # Discount time rate
ETA = 0.5 # Learning rate
MAX_STEPS = 200 # the Maximum number of steps per epsiode 
NUM_EPISODES = 1000 # the Maximum number of epsiodes
# Execute CartPole
env = gym.make(ENV) # Setting the task to execute
observation = env.reset() # initialize the ENV


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
    #writer = animation.FFMpegFileWriter(fps=1)
    anim.save('movie_cartpole.mp4') # The part where i save the animation
    display(display_animation(anim, default_mode = 'loop'))


# implement the Agent class
class Agent:
    '''It is a class that will act as a cartpole agent'''
    def __init__(self, num_states, num_actions):
        self.brain = Brain(num_states, num_actions) #Agent's brain role in determining behavior
        
    def update_Q_function(self, observation, action, reward, observation_next):
        '''Modifying the Q function'''
        self.brain.update_Q_table(observation, action, reward, observation_next)

    def get_action(self, observation, step):
        '''Action Determination'''
        action = self.brain.decide_action(observation, step)
        return action

# implement the brain class
class Brain:
    '''It is a class that will act as a agent's brain
    This class  actually performs Q learning'''
    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions # Get the number of actions(left, right)
        # Q table을 생성. 줄 수는 상태를 구간수^4(4는 변수의 수) 가지 값중 하나로 변환한 값,
        # 열수는 행동의 가짓수
        self.q_table = np.random.uniform(low=0,high=1, size=(NUM_DIZITIZED**num_states, num_actions))

    # Calculate the segment to make it discrete value
    def bins(self,clip_min, clip_max, num):
        '''Calculate the segment to convert continuous value to discrete value'''
        return np.linspace(clip_min, clip_max, num+1)[1:-1]
        # np.linspace(-2.4, 2.4, 6+1) = [-2.4, -1.6, -0.8, 0, 0.8, 1.6, 2.4]
        # indexing [1:-1] --> [-1.6, -0.8, 0, 0.8, 1.6]

    # A function of converting a continuous variable into a discrete variable
    # according to the interval value obtained by bins is implemented.
    def digitize_state(self,observation):
        '''convert the observation value to a discrete value'''
        cart_pos, cart_v, pole_angle, pole_v = observation

        digitized = [
            np.digitize(cart_pos,bins=self.bins(-2.4,2.4,NUM_DIZITIZED)),
            np.digitize(cart_v,bins=self.bins(-3.0,3.0,NUM_DIZITIZED)),
            np.digitize(pole_angle,bins=self.bins(-0.5,0.5,NUM_DIZITIZED)),
            np.digitize(pole_v,bins=self.bins(-2.0,2.0,NUM_DIZITIZED)),
            ]

        return sum([x * (NUM_DIZITIZED**i) for i, x in enumerate(digitized)])
        
    def update_Q_table(self, observation, action, reward, observation_next):
        ''' Q learning으로 Q table 수정'''
        state = self.digitize_state(observation) # 상태를 이산변수로 변환
        state_next = self.digitize_state(observation_next) #다음 상태를 이산 변수로 변환
        Max_Q_next = max(self.q_table[state_next][:])
        self.q_table[state, action] = self.q_table[state, action]+\
            ETA * (reward + GAMMA * Max_Q_next - self.q_table[state, action])
    
    def decide_action(self, observation, episode):
        ''' e-greedy 알고리즘을 적용해 서서히 최적행동의 비중을 늘림'''
        state = self.digitize_state(observation)
        epsilon = 0.5 * (1 / (episode+1))

        if epsilon <= np.random.uniform(0,1):
            action = np.argmax(self.q_table[state][:])
        else:
            action =  np.random.choice(self.num_actions) # 0,1 두가지 행동중 하나를 무작위로 선택
        
        return action

class Environment:
    '''CartPole을 실행하는 환경 역할을 하는 클래스'''
    def __init__(self):
        self.env = gym.make(ENV) # 실행할 태스크를 설정
        num_states = self.env.observation_space.shape[0] # 태스크의 상태 변수 수를 구함
        print("num_states:  " + str(num_states))
        num_actions = self.env.action_space.n # 가능한 행동 수를 구함
        print("num_actions:  " + str(num_actions))
        self.agent = Agent(num_states, num_actions) #에이전트 객체를 생성

    def run(self):
        '''실행'''
        complete_episodes = 0 # 성공한(195단계 이상 버틴) 에피소드 수
        is_episode_final = False # 마지막 에피소드 여부
        frames = [] # 애니메이션을 만드는데 사용할 이미지를 저장하는 변수

        for episode in range(NUM_EPISODES): #에피소드 수만큼 반복
            observation = self.env.reset() # 환경 초기화
            
            for step in range(MAX_STEPS): # 1에피소드에 해당하는 반복
                if is_episode_final is True: #마지막 에피소드이면 frames에 각 단계의 이미지를 저장
                    frames.append(self.env.render(mode='rgb_array'))
                # 행동을 선택
                action = self.agent.get_action(observation, episode)
                # 행동 a_t를 실행해 s_{t+1}, r_{t+1}을 계산
                observation_next,_, done,_ = self.env.step(action) # reward, info는 사용하지 않음으로
                                                                   # _로 처리함

                if done: # 200단계를 넘어서거나 일정 각도 이상 기울면 done의 값이 True가 됨
                    if step < 195:
                        reward = -1 #봉이 쓰러지면 패널티로 보상 -1부여
                        complete_episodes = 0 # 195 단계 이상 버티면 해당 에피소드를 성공 처리
                    else:
                        reward = 1 #쓰러지지 않고 에피소드를 끝내면 보상 1 부여
                        complete_episodes += 1 #에피소드 연속 성공 기록을 업데이트

                else:
                    reward = 0 # 에피소드 중에는 보상이 0
                
                # 다음 단계의 상태 observation_next로 Q함수를 수정
                self.agent.update_Q_function(observation, action, reward, observation_next)
                # 다음 단계 상태 관측
                observation = observation_next

                # 에피소드 마무리
                if done:
                    print('{0} Episode: Finished after {1} time steps'.format(episode, step+1))
                    break
            if is_episode_final is True: # 마지막 에피소드에서는 애니메이션을 만들고 저장
                display_frames_as_gif(frames)
                break
            if complete_episodes >= 10: # 10 에피소드 연속으로 성공한 경우
                print('10 에피소드 연속 성공')
                is_episode_final = True # 다음 에피소드가 마지막 에피소드가 됨

# main
cartpole_env = Environment()
cartpole_env.run()