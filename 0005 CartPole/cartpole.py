# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import gym

# A function that makes animations
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
        anim.save('movie_cartpole.mp4') # The part where i save the animation
        display(display_animation(anim, default_mode = 'loop'))

frames = []
env  = gym.make('CartPole-v0')
observation = env.reset() # 환경 초기화
print("observation =  ",observation)
for step in range(0,200):
        frames.append(env.render(mode='rgb_array')) # frames에 각 시각의 이미지를 추가함
        action = np.random.choice(2) #0(수레 왼쪽으로), 1(수레를 오른쪽으로) 두가지 행동을 무작위로 취함
        print("action = ", action)
        observation, reward, done, info  = env.step(action) # action 실행

#애니메이션을 파일로 저장하고 재생함
display_frames_as_gif(frames)
