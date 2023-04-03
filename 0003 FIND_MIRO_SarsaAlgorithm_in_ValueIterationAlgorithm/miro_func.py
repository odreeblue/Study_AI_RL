# -*- coding: utf-8 -*-
import numpy as np


## Calculate Random action policy pi_0
def simple_convert_into_pi_from_theta(theta):
    ''' Simple Ratio Calculation'''
    [m, n] = theta.shape # Get matrix size
    pi = np.zeros((m,n))
    for i in range(0,m):
        pi[i,:] = theta[i,:] / np.nansum(theta[i,:]) # Ratio Calculation
    
    pi = np.nan_to_num(pi) # Convert Nan to 0
    return pi

## Implement(구현하다) the e-greedy algorithm
def get_action(s, Q, epsilon, pi_0):
    direction = ["up","right","down","left"]
    
    # Decide the action
    if np.random.rand() < epsilon:
        # Determine a random action with probability " e "
        next_direction = np.random.choice(direction, p=pi_0[s,:])
    else:
        # Determine the action with the highest Q value
        next_direction = direction[np.nanargmax(Q[s,:])]

    # Convert the action to Index
    if next_direction == "up":
        action=0
    elif next_direction == "right":
        action=1
    elif next_direction == "down":
        action=2
    elif next_direction == "left":                           
        action=3
    
    return action

def get_s_next(s, a, Q, epsilon, pi_0):
    direction = ["up","right","down","left"]
    next_direction = direction[a] # direction of Action " a "

    # Determine the next state with Action
    if next_direction == "up":
        s_next = s-3 # When the agent moves up, its state value is reduced by 3
    elif next_direction == "right":
        s_next = s+1 # When the agent moves right, its state value is increased by 1
    elif next_direction == "down":
        s_next = s+3 # When the agent moves down, its state value is increased by 3
    elif next_direction == "left":                           
        s_next = s-1 # When the agent moves left, its state value is reduced by 1

    return s_next

## modify the action policy function Q with Sarsa Algorithm
def Sarsa(s, a, r, s_next, a_next, Q, eta, gamma):
    if s_next == 8:
        Q[s, a] = Q[s, a] +eta * (r - Q[s,a])
    else:
        Q[s, a] = Q[s, a] +eta * (r +gamma*Q[s_next, a_next] - Q[s,a])
    
    return Q

## function to get out of the maze with Sarsa algorithm, print state, action, history of Q value
def goal_maze_ret_s_a_Q(Q, epsilon, eta, gamma, pi):
    s = 0 # initial state
    a = a_next = get_action(s, Q, epsilon, pi) # first action
    s_a_histoty = [[0, np.nan]] # A list that records the history of the agent's action and state

    while (1): # Repeat until the agent reaches(도달하다) the target point(goal point)
        a = a_next # determine the action
        s_a_histoty[-1][1] = a # insert the current state to history
        s_next = get_s_next(s, a, Q, epsilon, pi) # get the state of the next step
        
        s_a_histoty.append([s_next, np.nan])# insert the next state to history, 
                                            # Action is Nan because we don't know yet
        
        if s_next == 8:
            r = 1 # A Reward is given if agent reaches the target point
            a_next = np.nan
        else:
            r = 0
            a_next = get_action(s_next, Q, epsilon, pi)
            # Calculate next action(a_next)

        # Modify the value function
        Q = Sarsa(s, a, r, s_next, a_next, Q, eta, gamma)

        # End decision
        if s_next == 8: # End if agent reches the target point
            break
        else:
            s = s_next
        
    return [s_a_histoty, Q]