# Visualization
import matplotlib.pyplot as plt
import os

# TrafficGen
import numpy as np
import math

# Memory
import random

# SetSumo Function
import sys
from sumolib import checkBinary

# ImportSettings Function
import configparser

class Visualization:
    def __init__(self,path,dpi):
        self.path=path
        self.dpi=dpi # Resolution in "Dots Per Inch"
        
    def DataAndPlot(self, data, filename, xlabel, ylabel):
        Min=min(data)
        Max=max(data)
        
        plt.rcParams.update({'font.size':20})
        
        plt.plot(data)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.margins(0)
        plt.ylim(Min-0.05*abs(Min), Max+0.05*abs(Max))
        
        figure=plt.gcf()
        figure.set_size_inches(20,12)
        figure.savefig(os.path.join(self.path, 'plot_'+filename+'.png'),dpi=self.dpi)
        plt.close("all")
        
        with open(os.path.join(self.path,'plot_'+filename+'_data.txt'), "w") as file:
            for value in data:
                file.write("%s\n"%value)

class TrafficGen:
    def __init__(self, MaxSteps, N_Cars):
        self.N_Cars=N_Cars
        self.MaxSteps=MaxSteps
        
    def GenerateRoutes(self, seed):
        # Generate route of cars every episode
        
        # Make tests reproducible
        np.random.seed(seed)
        
        # Cars generate according to weibull distribution (https://en.wikipedia.org/wiki/Weibull_distribution) and sort them
        Timing=np.sort(np.random.weibull(2,self.N_Cars))
        
        # Fit the distribution in the interval between 0 to MaxSteps
        CarGen=[]
        OldMin=math.floor(Timing[1])
        OldMax=math.floor(Timing[-1])
        NewMin=0
        NewMax=self.MaxSteps
        for x in Timing:
            CarGen=np.append(CarGen,((NewMax-NewMin)/(OldMax-OldMin)))
        
        #round every value to int, ie effective steps to determine when a car will be generated
        CarGen=np.rint(CarGen)
        
        
        # Make the file for car generation, each new line represents a new car
        with open("D:\Programming\Code\Python\RL\Traffic\TCS\Junction\EpisodeRoutes.rou.xml", "w") as route:
            print("""<routes>
            <vType accel="1.0" decel="4.5" id="Car" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" />

            <route id="W_N" edges="W2TL TL2N"/>
            <route id="W_E" edges="W2TL TL2E"/>
            <route id="W_S" edges="W2TL TL2S"/>
            <route id="N_W" edges="N2TL TL2W"/>
            <route id="N_E" edges="N2TL TL2E"/>
            <route id="N_S" edges="N2TL TL2S"/>
            <route id="E_W" edges="E2TL TL2W"/>
            <route id="E_N" edges="E2TL TL2N"/>
            <route id="E_S" edges="E2TL TL2S"/>
            <route id="S_W" edges="S2TL TL2W"/>
            <route id="S_N" edges="S2TL TL2N"/>
            <route id="S_E" edges="S2TL TL2E"/>""", file=route)

            
            for cc,step in enumerate(CarGen):
                StraightOrTurn = np.random.uniform()
                if StraightOrTurn < 0.73: # Cars go straight 73% of the time
                    rs=np.random.randint(1,5) # Helps choose which straight route a car should take
                    if rs==1:
                        print('    <vehicle id="W_E_%i" type="Car" route="W_E" depart="%s" departLane="random" departSpeed="10" />' % (cc, step), file=route)
                    elif rs ==2:
                        print('    <vehicle id="E_W_%i" type="Car" route="E_W" depart="%s" departLane="random" departSpeed="10" />' % (cc,step), file=route)
                    elif rs ==3:
                        print('    <vehicle id="N_S_%i" type="Car" route="N_S" depart="%s" departLane="random" departSpeed="10" />' % (cc,step), file=route)
                    else:
                        print('    <vehicle id="S_N_%i" type="Car" route="S_N" depart="%s" departLane="random" departSpeed="10" />' % (cc,step), file=route)
                        
                else: # Cars that take turns for the other 27% of the time
                    rt=np.random.randint(1,9) # Helps choose which route that has a turn a car should take
                    if rt==1:
                        print('    <vehicle id="W_N_%i" type="Car" route="W_N" depart="%s" departLane="random" departSpeed="10" />' % (cc,step), file=route)
                    elif rt ==2:
                        print('    <vehicle id="W_S_%i" type="Car" route="W_S" depart="%s" departLane="random" departSpeed="10" />' % (cc,step), file=route)                        
                    elif rt ==3:
                        print('    <vehicle id="N_W_%i" type="Car" route="N_W" depart="%s" departLane="random" departSpeed="10" />' % (cc,step), file=route)
                    elif rt ==4:
                        print('    <vehicle id="N_E_%i" type="Car" route="N_E" depart="%s" departLane="random" departSpeed="10" />' % (cc,step), file=route)
                    elif rt ==5:
                        print('    <vehicle id="E_N_%i" type="Car" route="E_N" depart="%s" departLane="random" departSpeed="10" />' % (cc,step), file=route)
                    elif rt ==6:
                        print('    <vehicle id="E_S_%i" type="Car" route="E_S" depart="%s" departLane="random" departSpeed="10" />' % (cc,step), file=route)
                    elif rt ==7:
                        print('    <vehicle id="S_W_%i" type="Car" route="S_W" depart="%s" departLane="random" departSpeed="10" />' % (cc,step), file=route)
                    elif rt ==8:
                        print('    <vehicle id="S_E_%i" type="Car" route="S_E" depart="%s" departLane="random" departSpeed="10" />' % (cc,step), file=route)
            
            print("</routes>", file=route)
            
class Memory:
    def __init__(self,SizeMax,SizeMin):
        self.Samples=[]
        self.SizeMax=SizeMax
        self.SizeMin=SizeMin
    
    def AddSample(self,Sample):
        self.Samples.append(Sample)
        if self.SizeNow()>self.SizeMax:
            # Removes oldest element when full
            self.Samples.pop(0)
    
    def GetSamples(self, N):
        if self.SizeNow()<self.SizeMin:
            return []
        
        if N>self.SizeNow():
            # Get all sampples
            return random.sample(self.Samples,self.SizeNow())
        else:
            # Returns 'BatchSize'(or N, in this case) number of samples
            return random.sample(self.Samples, N)
    
    def SizeNow(self):
        # Returns number of elements in Samples or how 'full' the memory is
        return len(self.Samples)
    
#Configure the various parameters of SUMO
def SetSumo(Gui, SumoCfgFileName,MaxSteps):
    if 'SUMO_HOME' in os.environ:
        Tools=os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(Tools)
    else:
        sys.exit("Please Declare Environment Variable 'SUMO_HOME'")
        
    if Gui == False:
        SumoBinary=checkBinary('sumo')
    else:
        SumoBinary=checkBinary('sumo-gui')
    
    # "sumoBinary" decides whether to use a GUI
    # "-c" loads the named config on startup
    # "os.path.join()" sets up the location of the sumo config file
    # "--no-step-logging" disables console output of current simulation step
    # "--waiting-time-memory" is the length of time interval, over which accumulated waiting time is taken into account
    # "str()" sets the maximum amount of steps allowed in the simulation
    SumoCmd=[SumoBinary,"-c",os.path.join('Junction',SumoCfgFileName),"--no-step-log","true","--waiting-time-memory",str(MaxSteps)]
    
    return SumoCmd

# Setting up the Directory for trained models and incrementally increase a variable to different models
def SetTrainPath(ModelPathName):
    ModelPath=os.path.join(os.getcwd(),ModelPathName,'')
    os.makedirs(os.path.dirname(ModelPath),exist_ok=True)
    
    Contents=os.listdir(ModelPath)
    if Contents:
        PreviousVersions=[int(Name.split("_")[1]) for Name in Contents]
        NewVersion=str(max(PreviousVersions)+1)
    else:
        NewVersion=1
    
    DataPath=os.path.join(ModelPath,'Model_'+str(NewVersion),'')
    os.makedirs(os.path.dirname(DataPath),exist_ok=True)
    
    return DataPath

# Setting up the path from which a trained model is taken
def SetTestPath(ModelNumber):
    ModelPath=os.path.join(os.getcwd(),"Models",'Model_'+str(ModelNumber),'')
    
    if os.path.isdir(ModelPath):
        PlotPath=os.path.join(ModelPath,'Testing','')
        os.makedirs(os.path.dirname(PlotPath),exist_ok=True)
        return ModelPath,PlotPath
    else:
        sys.exit('The Specified Model Doesnt Exist')
        
# Read Settings.ini
def ImportSettings(ConfigFile):
    Content=configparser.ConfigParser()
    Content.read(ConfigFile)
    Config={}
    
    Config['gui']=Content['Misc'].getboolean('gui')
    Config['maxsteps']=Content['Misc'].getint('maxsteps')
    Config['n_cars']=Content['Misc'].getint('n_cars')
        
    Config['numlayers']=Content['Model'].getint('numlayers')
    Config['layerwidth']=Content['Model'].getint('layerwidth')
    Config['batchsize']=Content['Model'].getint('batchsize')
    Config['learningrate']=Content['Model'].getfloat('learningrate')
    Config['numstates']=Content['Model'].getint('numstates')
    Config['numactions']=Content['Model'].getint('numactions')
    
    Config['greenduration']=Content['Simulation'].getint('greenduration')    
    Config['yellowduration']=Content['Simulation'].getint('yellowduration')
    Config['trainingepochs']=Content['Simulation'].getint('trainingepochs')
    
    Config['dpi']=Content['Visualisation'].getint('dpi')
    
    Config['maxmemorysize']=Content['Memory'].getint('maxmemorysize')
    Config['minmemorysize']=Content['Memory'].getint('minmemorysize')
       

    return Config

def MaxModelNumber(ModelPathName):
    ModelPath=os.path.join(os.getcwd(),ModelPathName,'')
    Contents=os.listdir(ModelPath)
    if Contents:
        PreviousVersions=[int(Name.split("_")[1]) for Name in Contents]
        VersionNumber=str(max(PreviousVersions)-1)
    else:
        VersionNumber=1
    
    return VersionNumber