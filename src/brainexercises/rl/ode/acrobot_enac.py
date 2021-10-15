#!/usr/bin/env python

try:
    ## following import success only when file is directly executed from command line
    ## otherwise will throw exception when executing as parameter for "python -m"
    # pylint: disable=W0611
    import __init__
except ImportError as error:
    ## when import fails then it means that the script was executed indirectly
    ## in this case __init__ is already loaded
    pass
 
import sys
from datetime import datetime
import argparse

# default backend GtkAgg does not plot properly on Ubuntu 8.04
import matplotlib
matplotlib.use('TkAgg')
 
import numpy
from scipy import mean
from matplotlib import pyplot as plt
 
# from pybrain.tools.example_tools import ExTools
from pybrain.rl.environments.ode import AcrobotEnvironment
from pybrain.rl.environments.ode.tasks import GradualRewardTask
from pybrain.rl.agents import LearningAgent
from pybrain.rl.learners import ENAC
# from pybrain.rl.learners import Reinforce
from pybrain.rl.experiments import EpisodicExperiment

from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules.tanhlayer import TanhLayer
 
from pybrain.tools.plotting import MultilinePlotter


## =========================================================================


## batch -- number of samples per gradient estimate (was: 20; more here due to stochastic setting)
def main( steps=200, batch=5, bias_unit=True, inner_layers=[], use_renderer=False, seed=None, alpha=0.1 ):
    start_time = datetime.now()
    
    usePlots = True
    # usePlots = False
    
    ## setting random seed
    if seed is None:
        seed = numpy.random.randint( 0, sys.maxint )
    numpy.random.seed( seed )
    print "random seed:", seed
    
    ## create task
    env = AcrobotEnvironment( renderer=use_renderer )
    env.render = False
    ## slower rendering
    env.dt = 0.019
    
    task = GradualRewardTask(env)
     
    ## create controller network
    layers = [ task.outdim ] + inner_layers + [ task.indim ]
    net = buildNetwork( *layers, bias=bias_unit, outputbias=bias_unit,
                        hiddenclass=TanhLayer, outclass=TanhLayer )

    print "network layers:", layers
    print "network:\n", net
    print "params size: %s" % len(net.params)
     
    ## create agent
    learner = ENAC()
    
    #### "Reinforce" seems to be bogus, because it does not learn and calculates "nan" in case of zeroed parameters.
    ## learner = Reinforce()
    
#     learner.gd.rprop = True
    if learner.gd.rprop is False:
        ## Back Prop variant
        learner.gd.alpha = alpha
# #         learner.gd.momentum = 0.9
#         learner.gd.momentum = 0.01

    print "learning params:", learner.gd.rprop, learner.gd.alpha, learner.gd.momentum
      
    agent = LearningAgent(net, learner)
    # agent = LearningAgent(net, learner)
    # agent.actaspg = False
     
    # create experiment
    experiment = EpisodicExperiment(task, agent)
    
    plotTitle = "seed:0x%X s:%s b:%s bias:%s hl:%s alpha:%s" % ( seed, steps, batch, bias_unit, inner_layers, alpha )
    print "metaparams:", plotTitle
    
    if usePlots:
        plt.ion()
        plt.figure()
        pl = MultilinePlotter(autoscale=1.2, xlim=[0, 50], ylim=[0, 1])
        pl.setLineStyle(linewidth=2)
        pl.setLabels( x="step", y="batch rewards mean", title=plotTitle )
     
    episodes = 0
    step = 0
    
    while step<steps:
    #while True:
        episodes += batch
        step += 1
        
        if use_renderer and (step % 10 == 0):
            ## demonstrate results
            prevRender = env.render
            env.render = True
            experiment.doEpisodes( 1 )
            env.render = prevRender
            agent.reset()
         
        experiment.doEpisodes( batch )
        reward = mean(agent.history.getSumOverSequences('reward'))
        
        agent.learn()
        agent.reset()
        
        if usePlots:
            pl.addData( 0, episodes, reward )
            pl.update()
            plt.pause(0.001)
     
#         print "params:\n%s" % agent.module.params
        print "step:", step, "reward: %s" % reward

    execution_time = datetime.now() - start_time

    print "duration: {}".format( execution_time )
    print "metaparams:", plotTitle
    print "done"
    
    if usePlots:
        plt.ioff()
        plt.show()


# ============================================================================


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Acrobot solution')
    parser.add_argument(        '--seed', action="store", type=int, default=None, help='RNG seed' )
    parser.add_argument( '-s',  '--steps', action="store", type=int, default=200, help='Steps' )
    parser.add_argument( '-b',  '--batch', action="store", type=int, default=10, help='Batch' )
    parser.add_argument(        '--bias', action="store", type=boolean_string, default=True, help='NN bias' )
    parser.add_argument( '-hl', '--hidden_layers', nargs='*', type=int, default=[4, 2], help='Define hidden layers' )
    parser.add_argument(        '--alpha', type=float, default=0.6, help='Learning rate' )
    parser.add_argument( '-r',  '--renderer', action="store_true", default=False, help='Use renderer' )
    
    args = parser.parse_args()
    
    main( args.steps, args.batch, args.bias, args.hidden_layers, args.renderer, seed=args.seed, alpha=args.alpha )
