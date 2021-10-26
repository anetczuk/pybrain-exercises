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
import os
from datetime import datetime
import argparse
import pickle
import time

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


SCRIPT_DIR = os.path.dirname( os.path.abspath(__file__) )
SRC_DIR    = os.path.abspath( os.path.join( SCRIPT_DIR, "..", "..", ".." ) )
TMP_DIR    = os.path.abspath( os.path.join( SRC_DIR, "..", "tmp" ) )


## =========================================================================


def process_events():
    ### code taken from "matplotlib.pyplot.pause()"
    interval = 0.01
    manager = plt._pylab_helpers.Gcf.get_active()
    if manager is not None:
        canvas = manager.canvas
        if canvas.figure.stale:
            canvas.draw_idle()
            ## 'show()' causes plot windows to blink/gain focus
#         show(block=False)
        canvas.start_event_loop(interval)
#     else:
#         time.sleep(interval)

    ## handle connections
    time.sleep(interval)


def load_input_data( input_file ):
    load_data = {}
    if input_file is not None:
        print "loading data from:", input_file
        if os.path.isfile(input_file):
            with open( input_file, 'rb' ) as fp:
                load_data = pickle.load( fp )
    return load_data


def save_data( data, output_file ):
    outPath = output_file
    if outPath is None:
        outPath = os.path.join( TMP_DIR, "current_enac.txt" )
#     print '\nstoring params:', data
    print 'storing params to:', outPath
    
    with open( outPath, 'wb' ) as fp:
        pickle.dump( data, fp )


def save_params( network, learner, task, output_file ):
#     data = {"network": nework}
#     save_data( data, output_file )

    learnerData = learner.dataset
    learner.dataset = None
    learnerLoglh = learner.loglh
    learner.loglh = None

    taskEnv = task.env
    task.env = None

#     data = { "network": network, "learner": learner, "task": task }
    data = { "network": network, "learner": learner }
    save_data( data, output_file )
     
    task.env = taskEnv
    learner.loglh = learnerLoglh
    learner.dataset = learnerData


def create_environment( use_renderer, learn, load_data, inner_layers, bias_unit, batch, alpha ):
    ## create task
    env = AcrobotEnvironment( renderer=use_renderer )
    if learn:
        env.render = False
    else:
        env.render = use_renderer
    ## slower rendering
    env.dt = 0.019
    
    task = load_data.get("task")
    if task is None:
        task = GradualRewardTask(env)
    else:
        task.env = env
    
    ## create controller network
    net = load_data.get("network")
    if net is None:
        layers = [ task.outdim ] + inner_layers + [ task.indim ]
        net = buildNetwork( *layers, bias=bias_unit, outputbias=bias_unit,
                            hiddenclass=TanhLayer, outclass=TanhLayer )

    ## create agent
    agent = None
    learner = None
    if learn:
        learner = load_data.get("learner")
        if learner is None:
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
    
    else:
        agent = LearningAgent(net, None)

    # create experiment
    experiment = EpisodicExperiment(task, agent)
    
    return (env, task, net, learner, agent, experiment)


def render_episode( env, experiment ):
    ##learner = experiment.optimizer
    
    ##oldBatch = learner.batchSize
    ##learner.batchSize = 1
    
    prevRender = env.render
    env.render = True
    experiment.doEpisodes( 1 )
    env.render = prevRender
    
    ##learner.batchSize = oldBatch
    
    
## ===========================================================================


## batch -- number of samples per gradient estimate (was: 20; more here due to stochastic setting)
def main( seed=None, steps=200, batch=5, bias_unit=True, inner_layers=[], alpha=0.1, use_renderer=False, learn=True,
          input_file=None, output_file=None ):
    
    start_time = datetime.now()
    
    usePlots = True
    # usePlots = False
    
    ## setting random seed
    if seed is None:
        seed = numpy.random.randint( 0, sys.maxint )
    numpy.random.seed( seed )
    print "random seed:", seed

    load_data = load_input_data( input_file )
    env, task, net, learner, agent, experiment = create_environment( use_renderer, learn, load_data, inner_layers, bias_unit, batch, alpha )
 
    biasLayer = bool( net["bias"] != None )
    layersDim = []    
    for layer in net.modulesSorted:
        layersDim.append( layer.dim )
    if biasLayer:
        layersDim.pop( 0 )

    print "network:\n", net
    print "network layers:", layersDim
    print "network params size: %s" % len(net.params)
    
    plotTitle = "seed:0x%X s:%s b:%s bias:%s l:%s alpha:%s" % ( seed, steps, batch, biasLayer, layersDim, alpha )
    
    print "metaparams:", plotTitle
    
    if usePlots:
#         plt.ion()
        plt.ioff()
        plt.figure()
        pl = MultilinePlotter(autoscale=1.2, xlim=[0, 50], ylim=[0, 1])
        pl.setLineStyle(linewidth=2)
        pl.setLabels( x="step", y="batch rewards mean", title=plotTitle )
        plt.show( block=False )

    try:
        episodes = 0
        step = 0
        
        while step<steps:
        #while True:
            episodes += batch
            step += 1
            
            reward = 0
            if learn:
                if use_renderer and (step % 50 == 0):
                    ## demonstrate results
                    render_episode( env, experiment )
                    agent.reset()
                 
                experiment.doEpisodes( batch )
                reward = mean(agent.history.getSumOverSequences('reward'))
                
                agent.learn()
                agent.reset()
            else:
                experiment.doEpisodes( batch )
                reward = mean(agent.history.getSumOverSequences('reward'))
                agent.reset()
            
            if usePlots:
                pl.addData( 0, step, reward )
                pl.update()
                process_events()
         
    #         print "params:\n%s" % agent.module.params
            print "step:", step, "reward: %s" % reward

        execution_time = datetime.now() - start_time

        print "duration: {}".format( execution_time )
        print "metaparams:", plotTitle
        print "done"
        
        if usePlots:
            plt.ioff()
            plt.show()

        if learn:
            save_params( net, learner, task, output_file )

    except KeyboardInterrupt:
        if learn:
            save_params( net, learner, task, output_file )

        if use_renderer:
            render_episode( env, experiment )


# ============================================================================


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Acrobot solution')
    parser.add_argument( '-if',  '--input_file', action="store", type=str, default=None, help='Input file containing learned parameters' )
    parser.add_argument( '-of',  '--output_file', action="store", type=str, default=None, help='Output file containing learned parameters' )
    parser.add_argument( '-iof', '--input_output_file', action="store", type=str, default=None, help='Input and output file containing learned parameters' )
    parser.add_argument(        '--learn', action="store", type=boolean_string, default=True, help='Should proceed learn phase?' )
    parser.add_argument(        '--seed', action="store", type=int, default=None, help='RNG seed' )
    parser.add_argument( '-s',  '--steps', action="store", type=int, default=200, help='Steps' )
    parser.add_argument( '-b',  '--batch', action="store", type=int, default=10, help='Batch' )
    parser.add_argument(        '--bias', action="store", type=boolean_string, default=True, help='NN bias' )
    parser.add_argument( '-hl', '--hidden_layers', nargs='*', type=int, default=[4, 2], help='Define hidden layers' )
    parser.add_argument(        '--alpha', type=float, default=0.6, help='Learning rate' )
    parser.add_argument( '-r',  '--renderer', action="store_true", default=False, help='Use renderer' )
    
    args = parser.parse_args()
    
    inputFile  = args.input_file
    if inputFile is None:
        inputFile = args.input_output_file
    outputFile = args.output_file
    if outputFile is None:
        outputFile = args.input_output_file
    
    main( seed=args.seed, steps=args.steps, batch=args.batch,
          bias_unit=args.bias, inner_layers=args.hidden_layers, alpha=args.alpha, 
          use_renderer=args.renderer, learn=args.learn,
          input_file=inputFile, output_file=outputFile 
        )
