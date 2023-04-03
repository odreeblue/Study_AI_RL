# -*- coding: utf-8 -*-
import numpy as np

# 정책 파라미터 theta를 행동정책 pi로 변환(소프트 맥스 함수 사용)하는 함수
def softmax_convert_into_pi_from_theta(theta):
    #비율 계산에 softmax함수 사용
    beta = 1.0
    
    [m, n]=theta.shape #theta의 행렬 크기를 구함
    pi = np.zeros((m,n))
    
    exp_theta = np.exp(beta*theta) # theta를 exp(theta)로 변환
    
    for i in range(0,m):
        # pi[i,:] = theta[i,:] / np.nansum(theta[i,:]) # 비율 계산
        pi[i,:] = exp_theta[i,:]/np.nansum(exp_theta[i,:])
        #softmax로 계산하는 코드
        
        #print(theta[i,:] / np.nansum(theta[i,:]))
    pi = np.nan_to_num(pi) # nan을 0으로 변환

    return pi


# 1단계 이동 후의 상태 s를 계산하는 함수
def get_action_and_next_s(pi,s):
    direction = ["up", "right", "down", "left"]

    next_direction = np.random.choice(direction, p=pi[s,:])
    #pi[s,:]의 확률에 따라, direction 값이 선택된다.

    if next_direction == "up":
        action=0
        s_next = s-10
    elif next_direction == "right":
        action=1
        s_next = s+1
    elif next_direction == "down":
        action=2
        s_next = s+10
    elif next_direction == "left":                           
        action=3
        s_next = s-1

    return [action, s_next]

# 목표 지점에 이를 때 까지 에이전트를 계속 이동시키는 함수

def goal_maze_ret_s_a(pi):
    s = 0 # 시작 지점
    s_a_history = [[0,np.nan]]

    while (1):
        [action,next_s] = get_action_and_next_s(pi,s)
        s_a_history[-1][1] = action
        
        s_a_history.append([next_s,np.nan])
        
        if next_s ==49:
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
                # 히스토리에서 상태 i인 것만 모아오는 리스트 컴프리핸션
                SA_ij = [SA for SA in s_a_history if SA==[i,j]]
                # 상태 i에서 행동 j를 취한 경우만 모음
                N_i=len(SA_i) # 상태 i에서 모든 행동을 취한 횟수
                N_ij = len(SA_ij) # 상태 i에서 행동 j를 취한 횟수

                delta_theta[i,j]=(N_ij - pi[i,j]*N_i) / T

    new_theta = theta + eta * delta_theta

    return new_theta
