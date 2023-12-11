# Handles the main loop that starts an episode on every iteration

import argparse
import configparser
import sys
import os
import datetime
import time
import tensorflow as tf
import sumo_visualizer as sv


from model import TrainModel
from utilities import SetSumo, SetTrainPath, Visualization, TrafficGen, Memory, ImportSettings, MaxModelNumber
from simulations import TrainingSimulation

from sumolib import checkBinary
def parse_args() -> argparse.Namespace:
    # Takes arguments from command line
    Parser=argparse.ArgumentParser()
    
    # Misc arguments
    Parser.add_argument('--Mode',help='Choose between making a new model and working further on an existing model',choices=['normal','retraining'],default='normal')
    Parser.add_argument('--Gui',help='GUI display option',type=bool,default=True)
    Parser.add_argument('--TotalEpisodes',help='Total Number of Episodes to train the model on',type=int,default=25)
    Parser.add_argument('--MaxSteps',help='Max Number of steps that can be taken',type=int,default=5400)
    Parser.add_argument('--N_Cars',help='Number of cars to be generated in each episode',type=int,default=1000)
    Parser.add_argument('--SaveSteps', help='Saves the model after every 5 episodes', action='store_true')
    
    # Model arguments
    Parser.add_argument('--NumLayers',help='Number of Layers in the Nueral Network',type=int,default=5)
    Parser.add_argument('--LayerWidth',help='Dimensionality of the Output Space',type=int,default=400)
    Parser.add_argument('--BatchSize',help='Size of Batch',type=int,default=100)
    Parser.add_argument('--LearningRate',help='Learning Rate for the Model',type=float,default=0.001)
    Parser.add_argument('--NumStates',help='Shape of the Inner Layers of the Nueral Network',type=int,default=80)
    Parser.add_argument('--NumActions',help='Output Shape of the Nueral Network',type=int,default=4)
    
    # Visualization arguments
    Parser.add_argument('--dpi',type=int,default=100)
    
    # Memory arguments
    Parser.add_argument('--MaxMemorySize',help='Maximum Size of Memory',type=int,default=50000)
    Parser.add_argument('--MinMemorySize',help='Minimum Size of Memory',type=int,default=600)
    
    # Training Simulation arguments
    Parser.add_argument('--GreenDuration',help='Duration in seconds for the traffic light to remain green',type=int,default=10)
    Parser.add_argument('--YellowDuration',help='Duration in seconds for the traffic light to remain yellow',type=int,default=4)
    Parser.add_argument('--TrainingEpochs',type=int,default=800)
    
    return Parser.parse_args()

if __name__ == "__main__":
    args=parse_args()
    config=configparser.ConfigParser()
    
    #Setting up cmd command to run sumo during simulation
    SumoCmd=SetSumo(args.Gui,"D:\Programming\Code\Python\RL\Traffic\TCS\Junction\SumoConfig.sumocfg",args.MaxSteps)
    
    # Setting up the Model Directory
    DataPath=SetTrainPath("D:\Programming\Code\Python\RL\Traffic\TCS\Models")
    
    Episode=0
    StartTimeStamp=datetime.datetime.now() # To Show the starting time when the program is done executing
    if(args.Mode=='normal'):
        
    # Setting up the .ini files to store the settings
        config.add_section('Misc')
        config.set('Misc','Gui',str(True))
        config.set('Misc','MaxSteps',str(args.MaxSteps))
        config.set('Misc','N_Cars',str(args.N_Cars))
        config.add_section('Model')
        config.set('Model','NumLayers',str(args.NumLayers))
        config.set('Model','LayerWidth',str(args.LayerWidth))
        config.set('Model','BatchSize',str(args.BatchSize))
        config.set('Model','LearningRate',str(args.LearningRate))
        config.set('Model','NumStates',str(args.NumStates))
        config.set('Model','NumActions',str(args.NumActions))
        config.add_section('Simulation')
        config.set('Simulation','GreenDuration',str(args.GreenDuration))
        config.set('Simulation','YellowDuration',str(args.YellowDuration))
        config.set('Simulation','TrainingEpochs',str(args.TrainingEpochs))
        config.add_section('Visualisation')
        config.set('Visualisation','dpi',str(args.dpi))
        config.add_section('Memory')
        config.set('Memory','MaxMemorySize',str(args.MaxMemorySize))
        config.set('Memory','MinMemorySize',str(args.MinMemorySize))
        FP=open(os.path.join(DataPath,"Settings.ini"),'x')
        config.write(FP)
        FP.close()

        # Initialising the Model
        Model=TrainModel(args.NumLayers,args.LayerWidth,args.BatchSize,args.LearningRate,args.NumStates,args.NumActions)
        
        # Graphs and Stuff
        Visualisation=Visualization(DataPath,args.dpi)
        
        #Generate Traffic/Routes taken by cars
        Traffic = TrafficGen(args.MaxSteps, args.N_Cars)
        
        #Creates Memory
        Memory=Memory(args.MaxMemorySize, args.MinMemorySize)
        
        # Creates the Env in which the model will be trained
        TrainingSimulation=TrainingSimulation(Model,Memory,Traffic,SumoCmd,0.75,args.MaxSteps,args.GreenDuration,args.YellowDuration,args.NumStates,args.NumActions,args.TrainingEpochs)
        
        while Episode<args.TotalEpisodes:
            print('=============== Episode',str(Episode+1), 'of', str(args.TotalEpisodes), ' ===============')
            Epsilon=1-(Episode/args.TotalEpisodes) # Sets epsilon for the current episode for epsilon greedy policy
            SimulationTime, TrainingTime = TrainingSimulation.RunTraining(Episode,Epsilon)
            print('\n=============== Episode Stats ===============')
            print('Simulation Time:', SimulationTime, 'Seconds')
            print('Training Time:', TrainingTime, 'Seconds')
            print('Total Time:', round(SimulationTime+TrainingTime,1), 'Seconds')
            print('=============================================')
            Episode+=1
            
            # import sumo_visualizer

            # # Create a SumoVisualizer object.
            # visualizer = sumo_visualizer.SumoVisualizer("D:\Programming\Code\Python\RL\Traffic\TCS\Junction\Environment.net.xml", "D:\Programming\Code\Python\RL\Traffic\TCS\Junction\EpisodeRoutes.rou.xml")

            # # Start the SUMO simulation.
            # visualizer.sumo_sim.start()

            # # Train the Deep Q Learning agent.
            # while not Model.is_trained():
            #     # Get the agent's state and action.
            #     agent_state = Model.get_state()
            #     agent_action = Model.act(agent_state)

            #     # Update the SUMO simulation with the agent's action.
            #     visualizer.sumo_sim.step(agent_action)

            #     # Visualize the SUMO simulation state.
            #     visualizer.visualize_step(agent_state, agent_action)

            # # Stop the SUMO simulation.
            # visualizer.close()
            
            if(args.SaveSteps and Episode%5==0 and Episode!=args.TotalEpisodes):
                StepData=DataPath+"Episode "+str(Episode)
                Viz=Visualization(StepData,args.dpi)
                Model.SaveModel(StepData)
                Viz.DataAndPlot(data=TrainingSimulation.RewardStore, filename='Reward', xlabel='Episode', ylabel='Total Negative Reward')
                Viz.DataAndPlot(data=TrainingSimulation.TotalWaitStore, filename='Delay', xlabel='Episode', ylabel='Total Delay (In Seconds)')
                Viz.DataAndPlot(data=TrainingSimulation.AverageQueueLengthStore, filename='Queue', xlabel='Episode', ylabel='Average Queue Length (Number Of Vehicles)') 
                print("Pausing the Training for 10 Minutes")
                #time.sleep(600)
                
        print('\n=============== Session Stats ===============')
        print('Start Time:', StartTimeStamp)
        print('End Time:', datetime.datetime.now())
        print('Model trained in this Session is saved at:', DataPath)
        print('=============================================')
        
        Model.SaveModel(DataPath)
        
        Visualisation.DataAndPlot(data=TrainingSimulation.RewardStore, filename='Reward', xlabel='Episode', ylabel='Total Negative Reward')
        Visualisation.DataAndPlot(data=TrainingSimulation.TotalWaitStore, filename='Delay', xlabel='Episode', ylabel='Total Delay (In Seconds)')
        Visualisation.DataAndPlot(data=TrainingSimulation.AverageQueueLengthStore, filename='Queue', xlabel='Episode', ylabel='Average Queue Length (Number Of Vehicles)') 

    
    elif(args.Mode=='retraining'):
        DataPath=os.path.join("Models","Model_"+str(MaxModelNumber("Models")))
        config=ImportSettings(os.path.join(DataPath,"Settings.ini"))
        Model=TrainModel(config['numlayers'],config['layerwidth'],config['batchsize'],config['learningrate'],config['numstates'],config['numactions'],tf.keras.models.load_model(os.path.join(DataPath,'TrainedModel.h5')))
        Visualisation=Visualization(DataPath,config['dpi'])
        Traffic=TrafficGen(config['maxsteps'],config['n_cars'])
        Memory=Memory(config['maxmemorysize'],config['minmemorysize'])
        TrainingSimulation=TrainingSimulation(Model,Memory,Traffic,SumoCmd,0.75,config['maxsteps'],config['greenduration'],config['yellowduration'],config['numstates'],config['numactions'],config['trainingepochs'])
        
        while Episode<args.TotalEpisodes:
            print('=============== Episode',str(Episode+1), 'of', str(args.TotalEpisodes), ' ===============')
            Epsilon=1-(Episode/args.TotalEpisodes) # Sets epsilon for the current episode for epsilon greedy policy
            SimulationTime, TrainingTime = TrainingSimulation.RunTraining(Episode,Epsilon)
            print('\n=============== Episode Stats ===============')
            print('Simulation Time:', SimulationTime, 'Seconds')
            print('Training Time:', TrainingTime, 'Seconds')
            print('Total Time:', round(SimulationTime+TrainingTime,1), 'Seconds')
            print('=============================================')
            Episode+=1
            
            if(args.SaveSteps and Episode%5==0 and Episode!=args.TotalEpisodes):
                StepData=DataPath+"Episode "+str(Episode)
                Viz=Visualization(StepData,config['dpi'])
                Model.SaveModel(StepData)
                Viz.DataAndPlot(data=TrainingSimulation.RewardStore, filename='Reward', xlabel='Episode', ylabel='Total Negative Reward')
                Viz.DataAndPlot(data=TrainingSimulation.TotalWaitStore, filename='Delay', xlabel='Episode', ylabel='Total Delay (In Seconds)')
                Viz.DataAndPlot(data=TrainingSimulation.AverageQueueLengthStore, filename='Queue', xlabel='Episode', ylabel='Average Queue Length (Number Of Vehicles)')
                print("Model Info is saved at:",StepData)
                print("Pausing the Training for 10 Minutes")
                time.sleep(600)
                
        print('\n=============== Session Stats ===============')
        print('Start Time:', StartTimeStamp)
        print('End Time:', datetime.datetime.now())
        print('Model trained in this Session is saved at:', DataPath)
        print('=============================================')      
        
        Model.SaveModel(DataPath)
        
        Visualisation.DataAndPlot(data=TrainingSimulation.RewardStore, filename='Reward', xlabel='Episode', ylabel='Total Negative Reward')
        Visualisation.DataAndPlot(data=TrainingSimulation.TotalWaitStore, filename='Delay', xlabel='Episode', ylabel='Total Delay (In Seconds)')
        Visualisation.DataAndPlot(data=TrainingSimulation.AverageQueueLengthStore, filename='Queue', xlabel='Episode', ylabel='Average Queue Length (Number Of Vehicles)')  