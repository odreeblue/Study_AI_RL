from Brain import Brain

class Agent:
        def __init__(self, num_states, num_actions,gamma):
                '''태스크의 상태 및 행동의 가짓수를 설정'''
                self.brain = Brain(num_states, num_actions, gamma) #Agent's brain role in determining behavior

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