import gym
from gym import error, spaces, utils
from gym.utils import seeding
from .simulator import skinTemperature
from .simulator import feedback
import numpy as np
import csv
import datetime
import eplus_env

# ref: https://github.com/openai/gym/tree/master/gym/envs
Ta_max = 30
Ta_min = 18
Rh_max = 51
Rh_min = 50
Tskin_max = 35.72
Tskin_min = 29.9


CLO = [0.4, 0.7, 1.0]
class simulationEnv(gym.Env):

	def __init__(self):
		self.nS = 1 # state space
		self.nA = 5 # action space 
		self.nO = 3 # number of occupant
		self.is_terminal = False
		self.is_eplus_terminal = True
		self.step_count = 0
		self.cur_Skin = []
		self.reward = []
		self.cur_Ta = 0
		self.cur_Rh = 0
		self.cur_setpoint = 0
		self.cur_MRT = 0 
		self.action = 0
		self.simulationTime = 0
		self.avg_reward = 0
		self.vote = feedback()
		self.Eplusenv = gym.make('Eplus-v0');
		self.actions  = []
			

	def _step(self, actions):
		""" take action and return next state and reward
		Parameters
		----------
		action: int, value is 0, 1, 2, 3, 4
			(0: temperature decrease 2 degreee; 1: decrease 1 degree; 2: no change, 
				3: increase 1 degree, 4: increase 2 degree)

		Return 
		----------
		ob:  array of state
		reward: float , PMV value

		"""
		state = []
		self.cur_Skin = []
		self.reward = []
		self.actions = []
		# get air temperature and air humidity after action
		#incr_Ta = action*0.5 - 0.5*int(self.nA/2)
		action = actions[0]
		# get the expected seperate setpoint from each agent
		for a in actions[1]:
			self.actions.append(self.cur_setpoint  + a - int(self.nA/2))
		

		incr_Ta = action - int(self.nA/2)
		self.action = action
		pre_Ta = self.cur_Ta
		pre_Rh = self.cur_Rh

		self.cur_setpoint = self.cur_setpoint + incr_Ta
		overflow = False
		if self.cur_setpoint > Ta_max:
			self.cur_setpoint = Ta_max
		elif self.cur_setpoint < Ta_min:
			self.cur_setpoint = Ta_min


		cursimTime, ob, self.is_terminal = self.Eplusenv.step([self.cur_setpoint,40])

		##only consider the situation when all occupants are there
		## if there is no occupant, heating setpoint is 1
		while ob[6] != 3:
			if self.is_terminal == True:
				break
			cursimTime, ob, self.is_terminal = self.Eplusenv.step([21,40])


		self.cur_Ta = ob[0]
		self.cur_Rh = ob[1]
		self.cur_MRT = ob[2]


		# get mean skin temperature from PierceSET model if there  occupants
		for i in range(self.nO):
			cur_Skin = skinTemperature().comfPierceSET(self.cur_Ta, self.cur_MRT, self.cur_Rh, CLO[i]) 

			self.cur_Skin.append(cur_Skin)
			if self.actions[i] > Ta_max:
				self.reward.append(-100)
			elif self.actions[i] < Ta_min:
				self.reward.append(-100)
			else:
				# get converted reward after action from PMV model
				self.reward.append(-1*self.vote.comfPMV(self.cur_Ta, self.cur_Ta, self.cur_Rh, CLO[i])[1])

			state.append(self._process_state_DDQN(cur_Skin))	

		
		self.avg_reward = sum(self.reward)/self.nO

		return state, self.reward, self.is_terminal, {}



	def _process_state_DDQN(self, skin_temp):
		""" convert skin temperature to value with 0 and 1
		Parameters
		----------
		skin_temp: float,  
		Return 
		----------
		state: float from 0 to 1

		""" 
		state = (skin_temp - Tskin_min)*1.0/(Tskin_max - Tskin_min) 
	
		return state



	def _reset(self):
		self.cur_setpoint = 21

		cursimTime, ob, self.is_terminal = self.Eplusenv.reset()
			

		cursimTime, ob, self.is_terminal = self.Eplusenv.step([self.cur_setpoint,40])
		while ob[6] != 3:
			if self.is_terminal == True:
				break
			cursimTime, ob, self.is_terminal = self.Eplusenv.step([self.cur_setpoint,40])
	
	
		self.cur_Ta = ob[0]
		self.cur_Rh = ob[1]
		self.cur_MRT = ob[2]
	

		state = []
		for i in range(self.nO):
			cur_Skin = skinTemperature().comfPierceSET(self.cur_Ta, self.cur_MRT, self.cur_Rh, CLO[i]) 
			self.cur_Skin.append(cur_Skin)
			state.append(self._process_state_DDQN(cur_Skin))
		return state



	def _render(self, mode='human', close=False):
		pass


	def my_render(self, folder, model='human', close=False):
	    with open(folder + "_render.csv", 'a', newline='') as csvfile:
	        fieldnames = ['time', 'simulationTime', 'action', 'setpoint', 'air_temp', 'air_humid', 'MRT', 'setpoint_0', 'setpoint_1',
	        'setpoint_2', 'skin_temp0', 'skin_temp1', 'skin_temp2', 'reward0', 'reward1', 'reward2', 
	        'avg_reward', 'is_terminal']
	        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
	        writer.writerow({fieldnames[0]: datetime.datetime.utcnow(), 
	        	fieldnames[1]:self.action, 
	        	fieldnames[2]:self.cur_setpoint,
				fieldnames[3]:self.cur_Ta, 
				fieldnames[4]:self.cur_Rh, 
				fieldnames[5]:self.cur_MRT, 
				fieldnames[6]:self.actions[0],
				fieldnames[7]:self.actions[1],
				fieldnames[8]:self.actions[2],
				fieldnames[9]:self.cur_Skin[0],
				fieldnames[10]:self.cur_Skin[1],
				fieldnames[11]:self.cur_Skin[2],
				fieldnames[12]:self.reward[0],
				fieldnames[13]:self.reward[1],
				fieldnames[14]:self.reward[2],
				fieldnames[15]:self.avg_reward,
				fieldnames[16]:self.is_terminal})




