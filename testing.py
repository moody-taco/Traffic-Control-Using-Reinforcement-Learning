from __future__ import absolute_import
from __future__ import print_function

import os
import argparse

from utilities import SetSumo,SetTestPath,TrafficGen,Visualization,ImportSettings
from model import TestModel
from simulations import TestingSimulation

def parse_args():
    # Takes arguments from command line
    Parser=argparse.ArgumentParser()
    Parser.add_argument('--ModelNumber',help='Model Number to be Tested',type=int,default = 4) #required=True)
    
    return Parser.parse_args()


if __name__=="__main__":
    args=parse_args()
    config=ImportSettings(os.path.join("Models","Model_4","Settings.ini"))
    # Setting up cmd command to run sumo during simulation
    SumoCmd=SetSumo(config['gui'],"SumoConfig.sumocfg",config['maxsteps'])
    # Setting up the path of the Model to be tested
    ModelPath,PlotPath=SetTestPath(args.ModelNumber)
    
    TestModel=TestModel(config['numstates'],ModelPath)
    TrafficGen=TrafficGen(config['maxsteps'],config['n_cars'])
    Visualization=Visualization(PlotPath,config['dpi'])
    TestingSimulation=TestingSimulation(TestModel,TrafficGen,SumoCmd,config['maxsteps'],config['greenduration'],config['yellowduration'],config['numstates'],config['numactions'])
    
    print('\n=============== Testing Episode ===============')
    SimulationTime=TestingSimulation.RunTesting(1000)
    print('Simulation Time:',SimulationTime,'Seconds')
    print("Model Testing Info is saved at:",PlotPath)
    
    Visualization.DataAndPlot(data=TestingSimulation.EpisodeReward, filename='Reward',xlabel='Action Step',ylabel='Reward')
    Visualization.DataAndPlot(data=TestingSimulation.EpisodeQueueLength, filename='Queue',xlabel='Step',ylabel='Queue Length (In Vehicles)')
    