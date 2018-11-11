import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import initializers
from keras import backend as K
from lib import plotting
from .simulator import skinTemperature
import sys
import csv

MAX_STEP = 1000
UPDATE_STEP = 30000


class QNAgent:
    def __init__(self, state_size, action_size,discount_factor, learning_rate):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=3000)
        self.gamma = discount_factor    # discount rate
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.step_count_total = 1
        self.target_update = UPDATE_STEP

    def _huber_loss(self, target, prediction):
        # sqrt(1+error^2)-1
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu')) #, kernel_initializer=initializers.RandomUniform(minval=-0.05, maxval=0))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def predict(self, state):
        act_values = self.model.predict(state)
        return act_values  # returns Q-value

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:

                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
 
            self.model.fit(state, target, epochs=1, verbose=0)
        ## update target model
        if self.step_count_total % self.target_update == 0:
            print('update')
            self.update_target_model()


    def load(self, name):
        self.model.load_weights(name)
        return self.model

    def save(self, name):
        self.model.save_weights(name)


def act(env, agents, states, epsilon, i_episode):
        if np.random.rand() <= epsilon:
            random_action = random.randrange(env.nA)
            return random_action, [random_action, random_action, random_action]
        i = 0
        act_values = []  
        best_action = []      
        for agent in agents:
            state = np.reshape(states[i], [1, env.nS])
            action_value = agent.predict(state)[0]
            act_values.append(action_value)
            best_action.append(np.argmax(action_value))
            # if(i_episode%100 == 0):
            #     print("agent" + str(i))
            #     print(action_value) 
            i = i + 1
        act_values = np.array(act_values)
        # get averaged Q - value
        act_values = np.average(act_values, axis=0)
        return np.argmax(act_values), best_action   # returns action


def q_learning(env, agents, num_episodes, batch_size, epsilon, epsilon_min, epsilon_decay, folder):
    """
    Q-Learning algorithm for fff-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Lambda time discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
        epsilon_decay: Each episode, epsilon is decayed by this factor
    
    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
  
    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))


    for i_episode in range(num_episodes):
        if epsilon > epsilon_min and i_episode > 2000:
            # complete random exploration 500 episodes, 
            # then decrase exploration till epsilon less than epsilon_min
            epsilon *= epsilon_decay
        sys.stdout.flush()

        states = env.reset()
   
       
        for t in range(MAX_STEP):
        
            ## Decide action
            action = act(env, agents, states, epsilon, i_episode)

            ## Advance the game to the next frame based on the action
            next_states, rewards, done, _ = env.step(action)

            env.my_render(folder)

            stats.episode_rewards[i_episode] += sum(rewards)/env.nO
            stats.episode_lengths[i_episode] = t+1

            if done: ## no need to remeber terminal state in this case
                break
            ## Remember the previous state, action, reward, and done
            for j in range(env.nO):
                states[j] = np.reshape(states[j], [1, env.nS])
                next_states[j] = np.reshape(next_states[j], [1, env.nS])
                agents[j].remember(states[j], action[0], rewards[j], next_states[j], done)   
                if len(agents[j].memory) > batch_size:
                    agents[j].replay(batch_size)  

            ## make next_state the new current state for the next frame.
            states = next_states

            
        mean_score = stats.episode_rewards[i_episode]/stats.episode_lengths[i_episode]
        print("episode: {}/{}, score: {}, e: {:.2}, steps:{}, mean score:{:.2}"
            .format(i_episode, num_episodes,  stats.episode_rewards[i_episode], epsilon, 
                stats.episode_lengths[i_episode], 
                 mean_score))
        #if(i_episode > 200):
        write_csv(folder, i_episode, stats.episode_lengths[i_episode], mean_score)
        if(i_episode%100 == 0):
            for j in range(env.nO):
                agents[j].save(folder + "_qn_O" + str(j) + "_" + str(i_episode) + ".h5")   
    for j in range(env.nO):
        agents[j].save(folder + "_qn-final_O" + str(j) + ".h5")           

    return stats


def write_csv(folder, episode, step_num, average_score):
    with open(folder + "_score.csv", 'a', newline='') as csvfile:
        fieldnames = ['episode', 'step_num', 'average_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({fieldnames[0]: episode, fieldnames[1]: step_num, 
            fieldnames[2]:average_score})


def evaluation(env, agent, folder):
    model = agent.load(folder + "_qn-final_O2.h5")
    for air in np.arange(18,30,0.5):
        state = skinTemperature().comfPierceSET(air, air-1, 35, 1.0) 
        print(state)
        state = env._process_state_DDQN(state)
        #env._print()
        state = np.reshape(state, [1, env.nS])
        target_f = model.predict(state)
        print(target_f)
        print(np.argmax(target_f))