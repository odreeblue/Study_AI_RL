# IMPORTING LIBRARIES

import sys

#import gym
from Env_Racing import Env1
#from Env_Racing import Env2
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import optimizers, losses
from tensorflow.keras import Model
from tensorflow.keras import layers

from IPython.display import clear_output

import time
class Network(Model):
    def __init__(self, state_size: int, action_size: int, 
    ):
        """Initialization."""
        super(Network, self).__init__()

        self.conv1 = layers.Conv2D(filters=16, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
        self.pool1 = layers.MaxPool2D(padding='same')
        self.conv2 = layers.Conv2D(filters=32, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
        self.pool2 = layers.MaxPool2D(padding='same')
        self.pool2_flat = layers.Flatten()
        self.dense3 = layers.Dense(units=32, activation=tf.nn.relu)
        self.dense4 = layers.Dense(units=16,activation=tf.nn.relu)
        self.dense5 = layers.Dense(units=4,activation=tf.nn.softmax)
    @tf.function
    def call(self, inputs_img, inputs_pos, training = False):
        net = self.conv1(inputs_img)
        net = self.pool1(net)
        net = self.conv2(net)
        net = self.pool2(net)
        net = self.pool2_flat(net)
        net = tf.keras.layers.Concatenate()([net,inputs_pos])
        net = self.dense3(net)
        net = self.dense4(net)
        net = self.dense5(net)
        return net
    
    def summary(self):
        x_img = layers.Input(shape=(64, 64, 1),name='image_input')
        x_pos = layers.Input(shape=(2,),name='position_input')
        model = Model(inputs=[x_img,x_pos], outputs=self.call(x_img,x_pos))
        return model.summary()

class DQNAgent:
    """A2CAgent interacting with environment.
        
    Attributes:
        env (gym.Env): openAI Gym environment
        gamma (float): discount factor
        entropy_weight (float): rate of weighting entropy into the loss function
        actor (tf.keras.Model): target actor model to select actions
        critic (tf.keras.Model): critic model to predict state values
        actor_optimizer (optim.Optimizer) : optimizer of actor
        critic_optimizer (optim.Optimizer) : optimizer of critic
        transition (list): temporory storage for the recent transition
        is_test (bool): flag to show the current mode (train / test)
    """

    def __init__(self, env: Env1,):
        """Initialization.
        
        Args:
            env (gym.Env): openAI Gym environment
            gamma (float): discount factor
        """
        
        # CREATING THE Q-Network
        self.env = env
        
        self.state_size = 2 # 태스크의 상태 변수 개수 : x position, z position
        self.action_size = 4 # 태스크의 행동 가짓 수 : 위, 아래, 오른쪽, 왼쪽
        
        self.lr = 7e-3
        self.gamma = 0.99    # discount rate
        self.model = Network(self.state_size, self.action_size)
        self.optimizers = optimizers.Adam(learning_rate=self.lr, )
        self.log_prob = None
    
    def get_action(self, state):
        #print("-------------------get_action---------------------------")
        #print("state : ", state)
        
        #prob = self.model(np.array([state]))
        prob = self.model(state['image'],state['position'])
        #print("prob = self.model(np.array([state])) -> ",prob)
        
        prob = prob.numpy()
        #print("prob = prob.numpy() -> ",prob)
        
        
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        #print("dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32) -> ",dist)
        
        
        action = dist.sample()
        #print("action = dist.sample() -> ",action)
        return int(action.numpy()[0])
    
    def actor_loss(self,prob, action, discnt_reward): 
        #print("action : ", action)
        #print("prob : ",prob)
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        #print("dist : ",dist)
        log_prob = dist.log_prob(action)
        #print("log_prob : ",log_prob)
        loss = -log_prob*discnt_reward
        return loss
    
    def n_step_td_target(self, states, actions, rewards, gamma):
        R_to_Go = 0
        discnt_rewards = []
        rewards.reverse()
        #print("rewards.shape :  ",len(rewards))
        #print("rewards : ",rewards)
        
        for r in rewards:
            R_to_Go = r + self.gamma*R_to_Go
            discnt_rewards.append(R_to_Go)
        discnt_rewards.reverse()
        #print("discnt_rewards :  ",discnt_rewards)
        st_img = []
        st_pos = []
        for st in states:
            st_img.append(st['image'].reshape((64,64,1)))
            st_pos.append(st['position'].reshape((2)))


        states_img  = np.array(st_img, dtype=np.float32)
        print("state_img size is ",states_img.shape)
        states_pos  = np.array(st_pos, dtype=np.float32)
        print("states_pos size is ",states_pos.shape)
        actions = np.array(actions, dtype=np.int32)
        discnt_rewards = np.array(discnt_rewards, dtype=np.float32)
        
        return states_img,states_pos, actions, discnt_rewards
        
    def train_step(self, states_img,states_pos, actions, discnt_rewards):
        discnt_rewards = tf.reshape(discnt_rewards, (len(discnt_rewards),))
        
        dqn_variable = self.model.trainable_variables
        #print("dqn_variable : ",dqn_variable)
        with tf.GradientTape() as tape:
            curr_Ps = self.model(states_img,states_pos, training=True)
            print("curr_Ps : ",curr_Ps)
            loss = self.actor_loss(curr_Ps, actions, discnt_rewards)
            print("loss : ", loss)
            
        dqn_grads = tape.gradient(loss, dqn_variable)
        #print("dqn_grads  : ",dqn_grads)
        #print(zip(dqn_grads,dqn_variable))
        self.optimizers.apply_gradients(zip(dqn_grads, dqn_variable))
        
        return loss


# CREATING THE ENVIRONMENT
miro_env = Env1()


# INITIALIZING THE Q-PARAMETERS
hidden_size = 64
max_episodes = 3000  # Set total number of episodes to train agent on.

# train
agent = DQNAgent(
    miro_env, 
#     memory_size, 
#     batch_size, 
#     epsilon_decay,
)

if __name__ == "__main__":
    #List to contain all the rewards of all the episodes given to the agent
    scores = []
    agent.env.connect() # 게임과 tcp/ip 연결

    #agent_x = 4.0
    #agent_y = -4.0
    
    complete_episodes = 0

    # EACH EPISODE    
    for episode in range(max_episodes):
        ## Reset environment and get first new observation
        time.sleep(2)
        initial_state, reward, done = agent.env.step(99)
        state = initial_state #[agent_x,agent_y] # 초기 위치
        episode_reward = 0
        done = False  # has the enviroment finished?
        
        states  = [] 
        actions = []
        rewards = []
            
        # EACH TIME STEP    
        while not done:
        # for step in range(max_steps):  # step index, maximum step is 200
            action = agent.get_action(state)
            
            # TAKING ACTION
            next_state, reward, done= agent.env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            # Our new state is state
            #state = next_state
            state = next_state
            episode_reward += reward
            #print(done)
            # if episode ends
            if done:
                scores.append(episode_reward)
                print("Episode " + str(episode+1) + ": " + str(episode_reward))
                
                states_img,states_pos, actions, discnt_rewards = agent.n_step_td_target(states, actions, rewards, 1)
                loss = agent.train_step(states_img,states_pos, actions, discnt_rewards) 
                break