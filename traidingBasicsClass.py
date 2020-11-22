import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts
from tf_agents.environments import py_environment


class StockAttackEnv(py_environment.PyEnvironment):

  def __init__(self):
    self._action_spec = array_spec.BoundedArraySpec(
        shape=(), dtype=np.int32, minimum=0, maximum=2, name='action')
    self._observation_spec = array_spec.BoundedArraySpec(
        shape=(5,), dtype=np.float64, minimum=0, name='observation')
    self._episode_ended        = False
    self.episodeTimeStepNumber = 0
    self.capital               = np.random.randint(0,2)
    self.ownedStock            = np.random.randint(0,2)
    self.lengthData            = 5
    self.data                  = np.ones(self.lengthData) * np.random.randint(0,2)  #data is random either all 1 or all 0
    self.changePosition        = np.random.randint(0, self.lengthData)
    #import pdb; pdb.set_trace()
    self.data[self.changePosition: self.lengthData] = np.ones(self.lengthData - self.changePosition) * np.random.randint(0,2) # data can either have a positive or negative jump, or no change at all
    self.cost                  = 0
    self.portfolioValue        = self.capital + self.ownedStock * self.data[0]

  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def _reset(self):
    self.capital               = np.random.randint(0,2)
    self.ownedStock            = np.random.randint(0,2)
    self.data                  = np.ones(self.lengthData) * np.random.randint(0,2)  #data is random either all 1 or all 0
    self.changePosition        = np.random.randint(0, self.lengthData)
    self.data[self.changePosition: self.lengthData] = np.ones(self.lengthData - self.changePosition) * np.random.randint(0,2) # data can either have a positive or negative jump, or no change at all
    self.cost                  = 0
    self.portfolioValue        = self.capital + self.ownedStock * self.data[0]
    self._episode_ended        = False
    self.episodeTimeStepNumber = 0
    self._observation_spec     = [[self.data[self.episodeTimeStepNumber], self.data[self.episodeTimeStepNumber]], 0, self.capital, self.ownedStock]
    return ts.restart(self._observation_spec)

  def _step(self, action):
    if self._episode_ended:
      return self.reset()

    price_buy  = self.data[self.episodeTimeStepNumber]+self.cost
    price_sell = self.data[self.episodeTimeStepNumber]-self.cost
    plausible  = True


    if action == 1:
        plausible = (self.ownedStock == 0) and (self.capital >= price_buy)
        if plausible:
            self.capital    -= price_buy
            self.ownedStock  = 1
    elif action == 0:
        pass
    elif action == 2:
        plausible = (self.ownedStock == 1)
        if plausible:
            self.capital   += price_sell * self.ownedStock
            self.ownedStock = 0
    else:
      raise ValueError('`action` should be 0 or 1.')


    self.portfolioValue = self.capital + self.ownedStock * price_sell
    plausbilityPenalty  = 0
    if not plausible:
        plausbilityPenalty = 1
    reward = self.portfolioValue - plausbilityPenalty


    self._action_spec = [action]
    if self.episodeTimeStepNumber >= 1:
        self._observation_spec = [self.data[self.episodeTimeStepNumber-1:self.episodeTimeStepNumber+1], plausbilityPenalty, self.capital, self.ownedStock]
    else:
        self._observation_spec = [[self.data[self.episodeTimeStepNumber], self.data[self.episodeTimeStepNumber]], plausbilityPenalty, self.capital, self.ownedStock]


    if self.data.shape[0]-1>self.episodeTimeStepNumber:
        self.episodeTimeStepNumber +=1
        #import pdb; pdb.set_trace()
        return ts.transition(self._observation_spec, reward, discount=0.0)
    else:
      self._episode_ended = True
      return ts.termination(self._observation_spec, reward)
