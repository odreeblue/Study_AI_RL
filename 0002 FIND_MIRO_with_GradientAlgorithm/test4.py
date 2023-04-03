# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import func as f
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


# 1. 정책 파라미터 theta_0를 행동정책 pi_0로 변환
pi_0 = f.softmax_convert_into_pi_from_theta(theta_0)

#print(pi_0)

stop_epsilon = 10**-4 # 정책의 변화가 10^-4보다 작아지면 학습 종료

theta = theta_0
pi = pi_0
is_continue = True

count = 1

# 1단계 이동 후의 상태 s를 계산하는 함수
def get_action_and_next_s(pi,s):
    direction = ["up", "right", "down", "left"]

    next_direction = np.random.choice(direction, p=pi[s,:])
    #pi[s,:]의 확률에 따라, direction 값이 선택된다.

    if next_direction == "up":
        action=0
        s_next = s-3
    elif next_direction == "right":
        action=1
        s_next = s+1
    elif next_direction == "down":
        action=2
        s_next = s+3
    elif next_direction == "left":                           
        action=3
        s_next = s-1

    return [action, s_next]
def goal_maze_ret_s_a(pi):
    s = 0 # 시작 지점
    s_a_history = [[0,np.nan]]

    while (1):
        [action,next_s] = get_action_and_next_s(pi,s)
        s_a_history[-1][1] = action
        
        s_a_history.append([next_s,np.nan])
        
        if next_s ==8:
            break
        else:
            s = next_s
        
    return s_a_history
def update_theta(theta, pi, s_a_history):
    eta = 0.1 # 학습률
    T = len(s_a_history)-1 # 목표 지점에 이르기까지 걸린 단계 수

    [m, n] = theta.shape # theta의 행렬 크기를 구함
    delta_theta = theta.copy() # del_theta 를 구할 준비, 포인터 참조를 피하기 위해 복제함

    # del_theta를 요소 단위로 계산
    for i in range(0,m):
        for j in range(0,n):
            if not(np.isnan(theta[i,j])): #theta가 nan이 아닌 경우
                SA_i = [SA for SA in s_a_history if SA[0]==i]
                #print("SA_i : ")
                #print(SA_i)
                # 히스토리에서 상태 i인 것만 모아오는 리스트 컴프리핸션
                SA_ij = [SA for SA in s_a_history if SA==[i,j]]
                #print("SA_ij : ")
                #print(SA_ij)
                # 상태 i에서 행동 j를 취한 경우만 모음
                N_i=len(SA_i) # 상태 i에서 모든 행동을 취한 횟수
                N_ij = len(SA_ij) # 상태 i에서 행동 j를 취한 횟수

                delta_theta[i,j]=(N_ij - pi[i,j]*N_i) / T

    new_theta = theta + eta * delta_theta

    return new_theta
while is_continue: # False 가 될 때까지 반복
    s_a_history = goal_maze_ret_s_a(pi) # 정책 pi를 따라 미로를 탐색한 히스토리를 구함
    #print(s_a_history)
    new_theta = update_theta(theta,pi,s_a_history) #파라미터 theta를 수정
    #print(new_theta)
    new_pi = f.softmax_convert_into_pi_from_theta(new_theta) # 정책 pi를 수정
    #print(new_pi)
    print(np.sum(np.abs(new_pi-pi))) # 정책의 변화를 출력
    print("목표 지점에 이르기까지 걸린 단계 수는 " +str(len(s_a_history)-1)+"단계입니다.")

    if np.sum(np.abs(new_pi-pi)) < stop_epsilon:
        is_continue = False
    else:
        theta = new_theta
        pi=new_pi
    #is_continue = False

'''
np.set_printoptions(precision=3, suppress= True) # 유효 자리수 3, 지수는 표시하지 않도록 설정
print(pi)



# 에이전트의 이동 과정을 시각화
# 참고 ~~ http://louistiao.me/posts/notebooks/embedding-matplotlib-animations-in-jupyter-notebooks/
from matplotlib import animation
#from IPython.display import HTML
def init():
    #배경 이미지 초기화
    line.set_data([],[])
    return (line,)

def animate(i):
    #프레임 단위로 이미지 생성
    state = s_a_history[i][0] #현재 위치
    x = (state % 3) + 0.5 # 상태의 x 좌표 : 3으로 나눈 나머지 +0.5
    y = 2.5 -int(state/3) # y 좌표 : 2.5에서 3으로 나눈 몫을 뺌
    line.set_data(x,y)
    return (line,)

anim = animation.FuncAnimation(fig,animate,init_func=init,frames=len(s_a_history),interval =200, repeat = False)
plt.show()
#HTML(anim.to_jshtml())

'''