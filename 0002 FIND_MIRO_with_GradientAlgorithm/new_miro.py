# -*- coding: utf-8 -*-
## this file is new miro. 
## it was written based on the test1~test6 files.

import numpy as np
import matplotlib.pyplot as plt
import func as f
# 초기상태의 미로 모습

# 전체 그림의 크기 및 그림을 나타내는 변수 선언
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

# 정책을 결정하는 파라미터의 초깃값 theta_0을 설정

# 줄은 상태 0~7, 열은 행동 방향(상 , 우 , 하, 좌 순)--> 시계 방향
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


# 1. 정책 파라미터 theta_0를 행동정책 pi_0로 변환
pi_0 = f.softmax_convert_into_pi_from_theta(theta_0)

print(pi_0)

stop_epsilon = 0.001 # 정책의 변화가 10^-4보다 작아지면 학습 종료

theta = theta_0
pi = pi_0

is_continue = True

count = 1

while is_continue: # False 가 될 때까지 반복
    s_a_history = f.goal_maze_ret_s_a(pi) # 정책 pi를 따라 미로를 탐색한 히스토리를 구함
    #print(s_a_history)
    new_theta = f.update_theta(theta,pi,s_a_history) #파라미터 theta를 수정
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
    x = (state % 10) + 0.5 # 상태의 x 좌표 : 3으로 나눈 나머지 +0.5
    y = 4.5 -int(state/10) # y 좌표 : 2.5에서 3으로 나눈 몫을 뺌
    line.set_data(x,y)
    return (line,)

anim = animation.FuncAnimation(fig,animate,init_func=init,frames=len(s_a_history),interval =200, repeat = False)
plt.show()
#print(s_a_history)
#HTML(anim.to_jshtml())
