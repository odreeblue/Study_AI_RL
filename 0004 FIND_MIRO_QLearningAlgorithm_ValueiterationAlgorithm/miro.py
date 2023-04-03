# -*- coding: utf-8 -*-
# 1. 패키지 로딩
from ensurepip import version
from matplotlib.colors import is_color_like
import numpy as np
import matplotlib.pyplot as plt
import miro_func as f

import sys
version_of_miro = sys.argv[1]
if len(sys.argv) != 2:
    print("Insufficient arguments")
    sys.exit()
# 작은 미로 : small, 큰 미로 : large
# 필요한 미로 버전을 파라미터에 넣어준다. ex : "python miro.py small" OR "python miro.py large"
print("version of miro : " + version_of_miro)


if version_of_miro == "small":
    # 2. 초기상태의 미로 모습
    ## 전체 그림의 크기 및 그림을 나타내는 변수 선언
    fig = plt.figure(figsize=(5,5))
    ax = plt.gca()
    ## 붉은 벽 그리기
    plt.plot([1,1],[0,1],color = 'red', linewidth = 2)
    plt.plot([1,2],[2,2],color = 'red', linewidth = 2)
    plt.plot([2,2],[2,1],color = 'red', linewidth = 2)
    plt.plot([2,3],[1,1],color = 'red', linewidth = 2)

    ##상태를 의미하는 문자열(S0-S8)표시
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

    ##S0에 녹샌 원으로 현재 위치를 표시
    line, = ax.plot([0.5],[2.5],marker="o",color='g',markersize=60)

    #3. 정책을 결정하는 파라미터의 초깃값 theta_0을 설정

    ## 줄은 상태 0~7, 열은 행동 방향(상 , 우 , 하, 좌 순)--> 시계 방향
    theta_0 = np.array([[np.nan,1,     1,     np.nan], #s0
                        [np.nan,1,     np.nan,1],      #s1
                        [np.nan,np.nan,1,     1],      #s2     
                        [1,     1,     1,     np.nan], #s3
                        [np.nan,np.nan,1,     1], #s4
                        [1,     np.nan,np.nan,np.nan], #s5
                        [1,     np.nan,np.nan,np.nan], #s6
                        [1,     1,     np.nan,np.nan], #s7
                        ]) #s8은 목표지점이므로 정책이 없다.
elif version_of_miro == "large":
    # 2. 초기상태의 미로 모습
    ## 전체 그림의 크기 및 그림을 나타내는 변수 선언
    fig = plt.figure(figsize=(10,5))
    ax = plt.gca()
    #붉은 벽 그리기
    #1열
    plt.plot([0,1],[2,2],color = 'red', linewidth = 2)
    #2열
    plt.plot([1,2],[4,4],color = 'red', linewidth = 2)
    plt.plot([1,1],[3,4],color = 'red', linewidth = 2)
    plt.plot([1,2],[3,3],color = 'red', linewidth = 2)
    plt.plot([1,2],[1,1],color = 'red', linewidth = 2)
    #3열
    plt.plot([2,3],[3,3],color = 'red', linewidth = 2)
    plt.plot([2,2],[2,3],color = 'red', linewidth = 2)
    plt.plot([2,3],[1,1],color = 'red', linewidth = 2)
    #4열
    plt.plot([3,3],[4,5],color = 'red', linewidth = 2)
    plt.plot([3,3],[3,4],color = 'red', linewidth = 2)
    plt.plot([3,4],[3,3],color = 'red', linewidth = 2)
    plt.plot([3,4],[2,2],color = 'red', linewidth = 2)
    plt.plot([3,3],[1,2],color = 'red', linewidth = 2)
    #5열
    plt.plot([4,5],[4,4],color = 'red', linewidth = 2)
    plt.plot([4,4],[2,3],color = 'red', linewidth = 2)
    plt.plot([4,4],[0,1],color = 'red', linewidth = 2)
    #6열
    plt.plot([5,5],[3,4],color = 'red', linewidth = 2)
    plt.plot([5,6],[3,3],color = 'red', linewidth = 2)
    plt.plot([5,5],[1,2],color = 'red', linewidth = 2)
    plt.plot([5,6],[1,1],color = 'red', linewidth = 2)
    #7열
    plt.plot([6,6],[4,5],color = 'red', linewidth = 2)
    plt.plot([6,6],[2,3],color = 'red', linewidth = 2)
    plt.plot([6,7],[2,2],color = 'red', linewidth = 2)
    plt.plot([6,7],[1,1],color = 'red', linewidth = 2)
    plt.plot([6,6],[0,1],color = 'red', linewidth = 2)
    #8열
    plt.plot([7,8],[4,4],color = 'red', linewidth = 2)
    plt.plot([7,7],[3,4],color = 'red', linewidth = 2)
    plt.plot([7,8],[2,2],color = 'red', linewidth = 2)
    plt.plot([7,7],[1,2],color = 'red', linewidth = 2)
    #9열
    plt.plot([8,9],[4,4],color = 'red', linewidth = 2)
    plt.plot([8,8],[3,4],color = 'red', linewidth = 2)
    plt.plot([8,8],[2,3],color = 'red', linewidth = 2)
    plt.plot([8,9],[1,1],color = 'red', linewidth = 2)
    #10열
    plt.plot([9,10],[3,3],color = 'red', linewidth = 2)
    plt.plot([9,9],[2,3],color = 'red', linewidth = 2)
    plt.plot([9,10],[1,1],color = 'red', linewidth = 2)
    ax.set_xlim(0,10)
    ax.set_ylim(0,5)

    #상태를 의미하는 문자열(S0-S8)표시
    plt.text(0.5,4.5,'S0',size=14,ha='center')
    plt.text(1.5,4.5,'S1',size=14,ha='center')
    plt.text(2.5,4.5,'S2',size=14,ha='center')
    plt.text(3.5,4.5,'S3',size=14,ha='center')
    plt.text(4.5,4.5,'S4',size=14,ha='center')
    plt.text(5.5,4.5,'S5',size=14,ha='center')
    plt.text(6.5,4.5,'S6',size=14,ha='center')
    plt.text(7.5,4.5,'S7',size=14,ha='center')
    plt.text(8.5,4.5,'S8',size=14,ha='center')
    plt.text(9.5,4.5,'S9',size=14,ha='center')

    plt.text(0.5,3.5,'S10',size=14,ha='center')
    plt.text(1.5,3.5,'S11',size=14,ha='center')
    plt.text(2.5,3.5,'S12',size=14,ha='center')
    plt.text(3.5,3.5,'S13',size=14,ha='center')
    plt.text(4.5,3.5,'S14',size=14,ha='center')
    plt.text(5.5,3.5,'S15',size=14,ha='center')
    plt.text(6.5,3.5,'S16',size=14,ha='center')
    plt.text(7.5,3.5,'S17',size=14,ha='center')
    plt.text(8.5,3.5,'S18',size=14,ha='center')
    plt.text(9.5,3.5,'S19',size=14,ha='center')

    plt.text(0.5,2.5,'S20',size=14,ha='center')
    plt.text(1.5,2.5,'S21',size=14,ha='center')
    plt.text(2.5,2.5,'S22',size=14,ha='center')
    plt.text(3.5,2.5,'S23',size=14,ha='center')
    plt.text(4.5,2.5,'S24',size=14,ha='center')
    plt.text(5.5,2.5,'S25',size=14,ha='center')
    plt.text(6.5,2.5,'S26',size=14,ha='center')
    plt.text(7.5,2.5,'S27',size=14,ha='center')
    plt.text(8.5,2.5,'S28',size=14,ha='center')
    plt.text(9.5,2.5,'S29',size=14,ha='center')

    plt.text(0.5,1.5,'S30',size=14,ha='center')
    plt.text(1.5,1.5,'S31',size=14,ha='center')
    plt.text(2.5,1.5,'S32',size=14,ha='center')
    plt.text(3.5,1.5,'S33',size=14,ha='center')
    plt.text(4.5,1.5,'S34',size=14,ha='center')
    plt.text(5.5,1.5,'S35',size=14,ha='center')
    plt.text(6.5,1.5,'S36',size=14,ha='center')
    plt.text(7.5,1.5,'S37',size=14,ha='center')
    plt.text(8.5,1.5,'S38',size=14,ha='center')
    plt.text(9.5,1.5,'S39',size=14,ha='center')

    plt.text(0.5,0.5,'S40',size=14,ha='center')
    plt.text(1.5,0.5,'S41',size=14,ha='center')
    plt.text(2.5,0.5,'S42',size=14,ha='center')
    plt.text(3.5,0.5,'S43',size=14,ha='center')
    plt.text(4.5,0.5,'S44',size=14,ha='center')
    plt.text(5.5,0.5,'S45',size=14,ha='center')
    plt.text(6.5,0.5,'S46',size=14,ha='center')
    plt.text(7.5,0.5,'S47',size=14,ha='center')
    plt.text(8.5,0.5,'S48',size=14,ha='center')
    plt.text(9.5,0.5,'S49',size=14,ha='center')

    plt.text(9.5,0.3,'GOAL',ha='center')

    plt.tick_params(axis = 'both', which = 'both',bottom=False,top=False,labelbottom=False,right=False,left=False,labelleft=False)
    #S0에 녹샌 원으로 현재 위치를 표시
    line, = ax.plot([0.5],[4.5],marker="o",color='g',markersize=30)

    #3. 정책을 결정하는 파라미터의 초깃값 theta_0을 설정

    ## 줄은 상태 0~7, 열은 행동 방향(상 , 우 , 하, 좌 순)--> 시계 방향
    theta_0 = np.array([
        [np.nan,1,1,np.nan],#S0
        [np.nan,1,np.nan,1],#S1
        [np.nan,np.nan,1,1],#S2
        [np.nan,1,1,np.nan],#S3
        [np.nan,1,np.nan,1],#S4
        [np.nan,np.nan,1,1],#S5
        [np.nan,1,1,np.nan],#S6
        [np.nan,1,np.nan,1],#S7
        [np.nan,1,np.nan,1],#S8
        [np.nan,np.nan,1,1],#S9
        [1,np.nan,1,np.nan],#S10
        [np.nan,1,np.nan,np.nan],#S11
        [1,np.nan,np.nan,1],#S12
        [1,1,np.nan,np.nan],#S13
        [np.nan,np.nan,1,1],#S14
        [1,1,np.nan,np.nan],#S15
        [1,np.nan,1,1],#S16
        [np.nan,np.nan,1,np.nan],#S17
        [np.nan,1,1,np.nan],#S18
        [1,np.nan,np.nan,1],#S19
        [1,1,np.nan,np.nan],#S20
        [np.nan,np.nan,1,1],#S21
        [np.nan,1,1,np.nan],#S22
        [np.nan,np.nan,np.nan,1],#S23
        [1,1,1,np.nan],#S24
        [np.nan,np.nan,1,1],#S25
        [1,1,np.nan,np.nan],#S26
        [1,np.nan,np.nan,1],#S27
        [1,np.nan,1,np.nan],#S28
        [np.nan,np.nan,1,np.nan],#S29
        [np.nan,1,1,np.nan],#S30
        [1,1,np.nan,1],#S31
        [1,np.nan,np.nan,1],#S32
        [np.nan,1,1,np.nan],#S33
        [1,np.nan,1,1],#S34
        [1,1,np.nan,np.nan],#S35
        [np.nan,np.nan,np.nan,1],#S36
        [np.nan,1,1,np.nan],#S37
        [1,1,np.nan,1],#S38
        [1,np.nan,np.nan,1],#S39
        [1,1,np.nan,np.nan],#S40
        [np.nan,1,np.nan,1],#S41
        [np.nan,1,np.nan,1],#S42
        [1,np.nan,np.nan,1],#S43
        [1,1,np.nan,np.nan],#S44
        [np.nan,np.nan,np.nan,1],#S45
        [np.nan,1,np.nan,np.nan],#S46
        [1,1,np.nan,1],#S47
        [np.nan,1,np.nan,1]#S48
        ])#S49 마지막은 도착지점이므로 정책없음

## The Initial state Of Action Value function
[a, b] = theta_0.shape # 열과 행의 개수를 변수 a, b 에 저장
                       # if small --> a = 8, b = 4
                       # if large --> a = 49, b = 4
Q = np.random.rand(a,b) * theta_0 *0.1 # Generate 8 x 4 OR 49 x 4 random number matrix
                                  # The reason for element-wise multiplication of theta_0 :
                                  # To give Nan to the action moving in the direction of the wall

## Calculate Random action policy pi_0
pi_0 = f.simple_convert_into_pi_from_theta(theta_0)

## Escape the maze with the Q_learning algorithm

eta = 0.1 # learning rate
gamma = 0.9 # time dicount rate
epsilon = 0.5 # epsion initial value of e-greedy algorithm 

v = np.nanmax(Q, axis = 1) # Calculate the maximum value of each state
is_continue = True
episode = 1

V = [] # 에피소드별로 상태가치를 저장
V.append(np.nanmax(Q, axis=1)) # 상태 별로 행동가치의 최대값을 계산

while is_continue: # Repeat until is_continue becomes False
    print("에피소드 : "+str(episode))

    # decrease the value of e little by little
    epsilon = epsilon / 2

    # After Escape the maze with the Sarsa algorithm, store action history and Q in variables
    [s_a_history, Q] = f.goal_maze_ret_s_a_Q(Q, epsilon, eta, gamma, pi_0, version_of_miro)

    # change in state value
    new_v = np.nanmax(Q,axis = 1) # Calculate maximum action value of each state
    print(np.sum(np.abs(new_v - v))) # print change in state value

    v = new_v
    V.append(v) #현재 에피소드가 끝난 시점의 상태가치 함수를 추가
    
    print(" 목표 지점에 이르기까지 걸린 단계 수는" + str(len(s_a_history)-1)+" 단계입니다.")

    # Repeat 100 OR 200 epsiode

    if version_of_miro == "small":
        max_episode = 100
    elif version_of_miro == "large":
        max_episode = 250

    episode = episode + 1
    if episode > max_episode:
        break

# 에이전트의 이동 과정을 시각화
from matplotlib import animation
def init():
    #배경 이미지 초기화
    line.set_data([],[])
    return (line,)
def animate_small(i):
    #프레임 단위로 이미지 생성
    state = s_a_history[i][0] #현재 위치
    x = (state % 3) + 0.5 # 상태의 x 좌표 : 3으로 나눈 나머지 +0.5
    y = 2.5 -int(state/3) # y 좌표 : 2.5에서 3으로 나눈 몫을 뺌
    line.set_data(x,y)
    return (line,)
def animate_large(i):
    #프레임 단위로 이미지 생성
    state = s_a_history[i][0] #현재 위치
    x = (state % 10) + 0.5 # 상태의 x 좌표 : 10으로 나눈 나머지 +0.5
    y = 4.5 -int(state/10) # y 좌표 : 4.5에서 10으로 나눈 몫을 뺌
    line.set_data(x,y)
    return (line,)
if version_of_miro == "small":
    anim = animation.FuncAnimation(fig,animate_small,init_func=init,frames=len(s_a_history),interval =200, repeat = False)
elif version_of_miro == "large":
    anim = animation.FuncAnimation(fig,animate_large,init_func=init,frames=len(s_a_history),interval =200, repeat = False)
plt.show()
