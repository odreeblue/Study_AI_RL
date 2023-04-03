#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# 초기상태의 미로 모습

# 전체 그림의 크기 및 그림을 나타내는 변수 선언
fig = plt.figure(figsize=(5,5))
ax = plt.gca()


#붉은 벽 그리기
plt.plot([1,1],[0,1],color = 'red', linewidth = 2)
plt.plot([1,2],[2,2],color = 'red', linewidth = 2)
plt.plot([2,2],[2,1],color = 'red', linewidth = 2)
plt.plot([2,3],[1,1],color = 'red', linewidth = 2)

#상태를 의미하는 문자열(S0-S8)표시
plt.text(0.5,2.5,'S0',size=14,ha='center')
plt.text(1.5,2.5,'S1',size=14,ha='center')
plt.text(2.5,2.5,'S2',size=14,ha='center')
plt.text(0.5,1.5,'S3',size=14,ha='center')
plt.text(1.5,1.5,'S4',size=14,ha='center')
plt.text(2.5,1.5,'S5',size=14,ha='center')
plt.text(0.5,0.5,'S6',size=14,ha='center')
plt.text(1.5,0.5,'S7',size=14,ha='center')
plt.text(2.5,0.5,'S8',size=14,ha='center')
plt.text(0.5,2.3,'START',ha='center')
plt.text(2.5,0.3,'GOAL',ha='center')

ax.set_xlim(0,3)
ax.set_ylim(0,3)

plt.tick_params(axis = 'both', which = 'both',bottom=False,top=False,labelbottom=False,right=False,left=False,labelleft=False)

#S0에 녹샌 원으로 현재 위치를 표시
line, = ax.plot([0.5],[2.5],marker="o",color='g',markersize=60)


# 정책을 결정하는 파라미터의 초깃값 theta_0을 설정

# 줄은 상태 0~7, 열은 행동 방향(상 , 우 , 하, 좌 순)--> 시계 방향
theta_0 = np.array([[np.nan,1,     1,     np.nan], #s0
                    [np.nan,1,     np.nan,1],      #s1
                    [np.nan,np.nan,1,     1],      #s2     
                    [1,     1,     1,     np.nan], #s3
                    [np.nan,np.nan,1,     1], #s4
                    [1,     np.nan,np.nan,np.nan], #s5
                    [1,     np.nan,np.nan,np.nan], #s6
                    [1,     1,     np.nan,np.nan], #s7
                    ]) #s8은 목표지점이므로 정책이 없다.


# 정책 파라미터 theta를 행동정책 pi로 변환하는 함수

def simple_convert_into_pi_from_theta(theta):
    '''단순히 값의 비율을 계산'''

    [m, n]=theta.shape #theta의 행렬 크기를 구함
    pi = np.zeros((m,n))
    for i in range(0,m):
        pi[i,:] = theta[i,:] / np.nansum(theta[i,:]) # 비율 계산
        print(theta[i,:] / np.nansum(theta[i,:]))
    pi = np.nan_to_num(pi) # nan을 0으로 변환

    return pi

pi_0 = simple_convert_into_pi_from_theta(theta_0)

# 1단계 이동 후의 상태 s를 계산하는 함수
def get_next_s(pi,s):
    direction = ["up", "right", "down", "left"]

    next_direction = np.random.choice(direction, p=pi[s,:])
    #pi[s,:]의 확률에 따라, direction 값이 선택된다.

    if next_direction == "up":
        s_next = s-3
    elif next_direction == "right":
        s_next = s+1
    elif next_direction == "down":
        s_next = s+3
    elif next_direction == "left":
        s_next = s-1

    return s_next

# 목표 지점에 이를 때 까지 에이전트를 계속 이동시키는 함수

def goal_maze(pi):
    s = 0 # 시작 지점
    state_history = [0]

    while (1):
        next_s = get_next_s(pi,s)
        state_history.append(next_s) # 경로 리스트에 다음 상태(위치)를 추가
        
        if next_s ==8:
            break
        else:
            s = next_s
        
    return state_history

state_history = goal_maze(pi_0)


print(state_history)
print("목표 지점에 이르기까지 걸린 단계 수는 "+str(len(state_history)-1)+"단계입니다.")


# 에이전트의 이동 과정을 시각화
# 참고 ~~ http://louistiao.me/posts/notebooks/embedding-matplotlib-animations-in-jupyter-notebooks/
from matplotlib import animation
#from IPython.display import HTML
def init():
    ''' 배경 이미지 초기화 '''
    line.set_data([],[])
    return (line,)

def animate(i):
    '''프레임 단위로 이미지 생성'''
    state = state_history[i] #현재 위치
    x = (state % 3) + 0.5 # 상태의 x 좌표 : 3으로 나눈 나머지 +0.5
    y = 2.5 -int(state/3) # y 좌표 : 2.5에서 3으로 나눈 몫을 뺌
    line.set_data(x,y)
    return (line,)

anim = animation.FuncAnimation(fig,animate,init_func=init,frames=len(state_history),interval =200, repeat = False)
plt.show()
#HTML(anim.to_jshtml())


