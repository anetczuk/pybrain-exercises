#!/usr/bin/env python


from __future__ import print_function

import numpy
from scipy import mean
from matplotlib import pyplot as plt

from pybrain.rl.environments.classic.mountaincar import MountainCar
from pybrain.rl.agents import LearningAgent
from pybrain.rl.learners import Reinforce, ENAC
from pybrain.rl.learners.valuebased import ActionValueNetwork, NFQ
from pybrain.rl.experiments import EpisodicExperiment
from pybrain.tools.shortcuts import buildNetwork


def create_bins_inf( minValue, maxValue, binNum ):
    valueRange = maxValue - minValue
    if binNum <= 1:
        return [ -numpy.inf, numpy.inf ]
    if binNum <= 2:
        return [ -numpy.inf, valueRange / 2 + minValue, numpy.inf ]
    stepsNum = binNum - 2
    step = valueRange / stepsNum
    next = minValue
    bins = [ -numpy.inf, minValue ]
    for x in range( stepsNum ):
        next += step
        bins.append( next )
    return bins + [ numpy.inf ]


##
## Convert multi dimensional state to 1-D state.
##
class StateConverter():

    def __init__(self, positionBins, speedBins):
        self.position_bounds = numpy.array( create_bins_inf(-1.4, 0.4, positionBins) )
        self.speed_bounds    = numpy.array( create_bins_inf(-7.0, 7.0, speedBins) )
        self.x_bounds        = numpy.array( [-numpy.inf, numpy.inf] )
        
        self.bounds = [ self.position_bounds, self.speed_bounds, self.x_bounds ]
    
        self.nbins_across_dims = [ len(self.position_bounds) - 1, len(self.speed_bounds) - 1, len(self.x_bounds) - 1 ]
        self.magic_array = numpy.cumprod([1] + self.nbins_across_dims)[:-1]
        
        self.numStates  = numpy.prod( self.nbins_across_dims )
        self.numActions = 3

    def getBinByList(self, state):
        bin_indices = []
        for stateIndex in range(0, len(state)):
            value = state[ stateIndex ]
            bound = self.bounds[ stateIndex ]
            digitized = numpy.digitize( [value], bound )[0] - 1
            bin_indices.append( digitized )
        return numpy.dot( self.magic_array, bin_indices )

    def getBinIndices(self, linear_index):
        """Given a linear index (integer between 0 and outdim), returns the bin
        indices for each of the state dimensions.
        """
        return linear_index / self.magic_array % self.nbins_across_dims


class NetworkDiscreteWrapper():
     
    def __init__( self, network ):
        self.network = network
     
    @property
    def indim(self):
        return self.network.indim
     
    @property
    def outdim(self):
#         return 1
        return self.network.outdim
     
    @property
    def paramdim(self):
        return self.network.paramdim
     
    @property
    def sequential(self):
        return self.network.sequential
     
    @property
    def name(self):
        return self.network.name
     
    @property
    def params(self):
        return self.network.params
     
    def _getParameters(self):
        return self.network._getParameters()
     
    def _setParameters(self, p, owner=None):
        return self.network._setParameters( p, None )
     
    @property
    def derivs(self):
        return self.network.derivs
     
    def _getDerivatives(self):
        return self.network._getDerivatives()
     
    def _setDerivatives(self, d, owner=None):
        return self.network._setDerivatives( d, None )
     
    def _resetBuffers(self, length=1):
        return self.network._resetBuffers( length )
    
    def reset(self):
        return self.network.reset()
    
    def activate(self, obst):
        return self.network.activate( obst )

    @property
    def outputerror(self):
        return self.network.outputerror
     
    @property
    def outputbuffer(self):
        return self.network.outputbuffer
    
    def backward(self):
        return self.network.backward()
     
    @property
    def inputerror(self):
        return self.network.inputerror
     
     
    
class IntegerLearningAgent( LearningAgent ):
    
#     def __init__(self, stateConverter, module, learner = None):
#         LearningAgent.__init__(self, module, learner)
#         self.converter = stateConverter
#     
#     def integrateObservation(self, obs):
#         bin = self.converter.getBinByList( obs )
#         return LearningAgent.integrateObservation( self, [bin] )

#     def giveReward(self, reward):
#         reward = reward * 2 - 1
#         LearningAgent.giveReward( self, reward )

    def getAction(self):
        action = LearningAgent.getAction(self)
#         return action.astype(int)
        index_max = max(range(len(action)), key=action.__getitem__)
        return index_max


## ============================================================================


numpy.random.seed( 112 )
# numpy.random.seed( 114 )


plotProgress = True
# plotProgress = False


# create task
## accepts one of following actions:
##    0 means -1.0 force
##    1 means  0.0 force
##    2 means  1.0 force
task = MountainCar()
 

# binsNum = 8
# stateConverter = StateConverter( binsNum, binsNum )

## create value table and initialize with ones
net = buildNetwork( 3, 3, 3, bias=True )
# net._setParameters( numpy.ones( len(net.params) ) )
# module = NetworkDiscreteWrapper( net )
module = net

# learner = Reinforce()
learner = ENAC()
# learner.gd.rprop = True
# only relevant for RP
learner.gd.deltamin = 0.0001
#agent.learner.gd.deltanull = 0.05
# only relevant for BP
learner.gd.alpha = 0.1
learner.gd.momentum = 0.9

# print( "learning rate:", learner.learningRate )

# create agent
agent = IntegerLearningAgent( module, learner )
# agent = LearningAgent( module, learner )

## create experiment
experiment = EpisodicExperiment( task, agent )


if plotProgress:
    performance = []
    
    plt.ion()
    
    def plotPerformance(values, fig):
        plt.figure(fig.number)
        plt.clf()
        plt.plot(values, 'o-')
        plt.gcf().canvas.draw()
        ## Without the next line, the pyplot plot won't actually show up.
        plt.pause(0.001)

    pf_fig = plt.figure()


EPISODES = 300
    
for ep in range(1, EPISODES):
    ## execute and learn
    rewards = experiment.doEpisodes( 1 )
    agent.learn()
    agent.reset()
    
    print( "net:\n", net.params )
    
    ## plot data
    meanReward = mean([mean(x) for x in rewards])

    steps = len(rewards[0])
 
    if plotProgress:
        performance.append( meanReward )
        #     performance.append( steps )
        if ep % 20 == 0:
            plotPerformance( performance, pf_fig )

    print( "episode:", ep, "mean reward:", meanReward, "steps:", steps )


if plotProgress:
    plotPerformance( performance, pf_fig )
    plt.show( block=True )
