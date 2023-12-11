import traci
import numpy as np
import random
import timeit

#phase codes for traffic lights
NS_Green=0
NS_Yellow=1
NSL_Green=2
NSL_Yellow=3
EW_Green=4
EW_Yellow=5
EWL_Green=6
EWL_Yellow=7

class TrainingSimulation:
    # Basic Class Definition
    def __init__(self, Model, Memory, Traffic, Sumo, Gamma, MaxSteps, GreenDuration, YellowDuration, NumStates, NumActions, TrainingEpochs):
        self.Model=Model
        self.Memory = Memory
        self.Traffic = Traffic
        self.Gamma = Gamma
        self.Step = 0
        self.Sumo = Sumo
        self.MaxSteps = MaxSteps
        self.GreenDuration = GreenDuration
        self.YellowDuration = YellowDuration
        self.NumStates = NumStates
        self.NumActions = NumActions
        self.RewardStore = []
        self.TotalWaitStore = []
        self.AverageQueueLengthStore = []
        self.TrainingEpochs = TrainingEpochs
    
    # Runs an episode of simuation and then trains the model on the generated simulation
    def RunTraining(self, episode, epsilon):
        StartTime=timeit.default_timer()
        
        # Generate route file
        self.Traffic.GenerateRoutes(seed=episode)
        
        # Setting up Sumo
        traci.start(self.Sumo)
        
        print("=====Simulating Cars=====")
        
        #initializations
        self.Step=0
        self.WaitingTimes={}
        self.SumNegativeReward=0
        self.SumQueueReward=0
        self.SumWaitingTime=0
        OldTotalWait=0
        OldState=-1
        OldAction=-1
        
        while self.Step<self.MaxSteps:
            # Current State of the Junction
            CurrentState=self.GetState()
            
            # Waiting time is the number of seconds that a car has waiting after being spawned into the env
            # Here we will be getting the sum of all waiting times for the cars in all lanes
            CurrentTotalWait=self.CollectWaitingTimes()
            Reward=OldTotalWait-CurrentTotalWait
            
            # Save Data to Memory
            if self.Step!=0:
                self.Memory.AddSample((OldState,OldAction,Reward,CurrentState))
            
            # Choose an Action or Light Phase to activate based on the current state of the junction
            Action=self.ChooseAction(CurrentState, epsilon)
            
            # If the chosen action/phase is different than the old one, change to yellow phase
            if self.Step!=0 and OldAction!=Action:
                self.SetYellowPhase(OldAction)
                self.Simulate(self.YellowDuration)
                
            # Execute the chosen phase
            self.SetGreenPhase(Action)
            self.Simulate(self.GreenDuration)
            
            # Save Variables for next iteration and Collect Reward
            OldState=CurrentState
            OldAction=Action
            OldTotalWait=CurrentTotalWait
            
            # Save only meaningful rewards to see if the agent is behaving to as planned
            if Reward<0:
                self.SumNegativeReward+=Reward
                
        self.SaveEpisodeStats()
        print("Total Reward: ", self.SumNegativeReward, "| Epsilon: ", round(epsilon, 2))
        traci.close()
        SimulationTime=round(timeit.default_timer() - StartTime, 1)
        
        print("=====Training The Model=====")
        StartTime=timeit.default_timer()
        for _ in range(self.TrainingEpochs):
            self.Replay()
        TrainingTime=round(timeit.default_timer() - StartTime, 1)
        
        return SimulationTime, TrainingTime

    # Gathers states while executing steps in sumo
    def Simulate(self, StepsTodo):
        if(self.Step+StepsTodo)>=self.MaxSteps:
            StepsTodo=self.MaxSteps-self.Step
            
        while StepsTodo>0:
            traci.simulationStep() # Simulate 1 step
            self.Step+=1
            StepsTodo-=1
            QueueLength=self.GetQueueLength()
            self.SumQueueReward+=QueueLength
            self.SumWaitingTime+=QueueLength # 1 one step while waiting in queue

    # Collect the waiting time for every car in the Incoming Roads            
    def CollectWaitingTimes(self):
        IncomingRoads=["E2TL", "N2TL", "W2TL", "S2TL"]
        CarList=traci.vehicle.getIDList()
        for CarID in CarList:
            WaitTime=traci.vehicle.getAccumulatedWaitingTime(CarID)
            RoadID=traci.vehicle.getRoadID(CarID) # Get Road ID on which the vehicle is
            if RoadID in IncomingRoads: # Considers only waiting time of cars in incoming roads
                self.WaitingTimes[CarID]=WaitTime
            else:
                if CarID in self.WaitingTimes: # Car that was tracked and has now cleared the intersection
                    del self.WaitingTimes[CarID]
        TotalWaitingTime=sum(self.WaitingTimes.values())
        
        return TotalWaitingTime
    
    # Decide whether to explore or exploit, according to an epsilon-greedy policy
    def ChooseAction(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.NumActions -1)
        else:
            return np.argmax(self.Model.PredictOne(state))

    # Activates the green correct green light combination
    def SetYellowPhase(self, OldAction):
        YellowPhaseCode=OldAction*2+1
        traci.trafficlight.setPhase("TL", YellowPhaseCode)

    # Activates the green correct green light combination
    def SetGreenPhase(self, ActionNumber):
        if ActionNumber==0:
            traci.trafficlight.setPhase("TL",NS_Green)
        elif ActionNumber==1:
            traci.trafficlight.setPhase("TL",NSL_Green)
        elif ActionNumber==2:
            traci.trafficlight.setPhase("TL",EW_Green)
        elif ActionNumber==3:
            traci.trafficlight.setPhase("TL",EWL_Green)

    # Get the number of cars with no speed in all the incoming lanes
    def GetQueueLength(self):
        HaltN=traci.edge.getLastStepHaltingNumber("N2TL")
        HaltS=traci.edge.getLastStepHaltingNumber("S2TL")
        HaltE=traci.edge.getLastStepHaltingNumber("E2TL")
        HaltW=traci.edge.getLastStepHaltingNumber("W2TL")
        QueueLength=HaltN+HaltE+HaltS+HaltW
        
        return QueueLength
    
    # Retrieves the state of the junction from sumo
    def GetState(self):
        State=np.zeros(self.NumStates)
        CarList=traci.vehicle.getIDList()
        
        for CarID in CarList:
            LanePosition=traci.vehicle.getLanePosition(CarID)
            LaneID=traci.vehicle.getLaneID(CarID)
            LanePosition=750-LanePosition # If the car is close to the traffic light
            
            # Distance from the Traffic Light being mapped into cells
            if LanePosition < 7:
                LaneCell = 0
            elif LanePosition < 14:
                LaneCell = 1
            elif LanePosition < 21:
                LaneCell = 2
            elif LanePosition < 28:
                LaneCell = 3
            elif LanePosition < 40:
                LaneCell = 4
            elif LanePosition < 60:
                LaneCell = 5
            elif LanePosition < 100:
                LaneCell = 6
            elif LanePosition < 160:
                LaneCell = 7
            elif LanePosition < 400:
                LaneCell = 8
            elif LanePosition <= 750:
                LaneCell = 9
            
            # Finding where the car is located
            # Here W2TL_3, N2TL_3, E2TL_3, S2TL_3 are 'left-only' turns
            if LaneID == "W2TL_0" or LaneID == "W2TL_1" or LaneID == "W2TL_2":
                LaneGroup = 0
            elif LaneID == "W2TL_3":
                LaneGroup = 1
            elif LaneID == "N2TL_0" or LaneID == "N2TL_1" or LaneID == "N2TL_2":
                LaneGroup = 2
            elif LaneID == "N2TL_3":
                LaneGroup = 3
            elif LaneID == "E2TL_0" or LaneID == "E2TL_1" or LaneID == "E2TL_2":
                LaneGroup = 4
            elif LaneID == "E2TL_3":
                LaneGroup = 5
            elif LaneID == "S2TL_0" or LaneID == "S2TL_1" or LaneID == "S2TL_2":
                LaneGroup = 6
            elif LaneID == "S2TL_3":
                LaneGroup = 7
            else:
                LaneGroup = -1
                
            if LaneGroup>=1 and LaneGroup<=7:
                CarPosition=int(str(LaneGroup)+str(LaneCell)) # Creates a number between 0 and 79
                ValidCar=True
            elif LaneGroup==0:
                CarPosition=LaneCell
                ValidCar=True
            else:
                ValidCar=False # Flag to make sure the cars crossing the intersection/driving away from the intersection arent detected
            
            if ValidCar:
                State[CarPosition]=1 # Writes the position of the Car in the state array as 'cell occupied'
        
        return State
    
    # Retrieve a group of samples from memory and update the learning equation for each of them, then train the Nueral Network
    def Replay(self):
        Batch=self.Memory.GetSamples(self.Model.BatchSize)
        if len(Batch)>0:
            States=np.array([Val[0] for Val in Batch])
            NextStates=np.array([Val[3] for Val in Batch])
            
            # Predictions of Q(State) and Q(NextState) for every sample
            QSA=self.Model.PredictBatch(States)
            QSAD=self.Model.PredictBatch(NextStates)
            
            # Setting up training arrays
            x=np.zeros((len(Batch), self.NumStates))
            y=np.zeros((len(Batch), self.NumActions))
            
            for i, b in enumerate(Batch):
                State, Action, Reward, _=b[0], b[1], b[2], b[3]
                CurrentQ=QSA[i]
                CurrentQ[Action]=Reward+self.Gamma*np.amax(QSAD[i]) # Updates Q(State,Action)
                x[i]=State
                y[i]=CurrentQ
        
            self.Model.TrainBatch(x,y)

    # Save stats of the episode to plot
    def SaveEpisodeStats(self):
        self.RewardStore.append(self.SumNegativeReward)
        self.TotalWaitStore.append(self.SumWaitingTime)
        self.AverageQueueLengthStore.append(self.SumQueueReward / self.MaxSteps)
    
class TestingSimulation:
    # Basic Class Definition
    def __init__(self, Model, Traffic, Sumo, MaxSteps, GreenDuration, YellowDuration, NumStates, NumActions):
        self.Model=Model
        self.Traffic=Traffic
        self.Step=0
        self.Sumo=Sumo
        self.MaxSteps=MaxSteps
        self.GreenDuration=GreenDuration
        self.YellowDuration=YellowDuration
        self.NumStates=NumStates
        self.NumActions=NumActions
        self.EpisodeReward=[]
        self.EpisodeQueueLength=[]

    # Runs the testing simulation        
    def RunTesting(self, episode):
        StartTime=timeit.default_timer()
        
        self.Traffic.GenerateRoutes(seed=episode)
        traci.start(self.Sumo)
        print("=====Testing Cars=====")
        
        # Initialisations
        self.Step=0
        self.WaitingTimes={}
        OldWaitTime=0
        OldAction=-1 # Arbitrary Initialisation
        
        while self.Step<self.MaxSteps:
            CurrentState=self.GetState()
            CurrentTotalWait=self.CollectWaitingTimes()
            Reward=OldWaitTime-CurrentTotalWait
            
            Action=self.ChooseAction(CurrentState)
            
            if self.Step!=0 and OldAction!=Action:
                self.SetYellowPhase(OldAction)
                self.Simulate(self.YellowDuration)
            
            self.SetGreenPhase(Action)
            self.Simulate(self.GreenDuration)
            
            OldAction=Action
            OldWaitTime=CurrentTotalWait
            self.EpisodeReward.append(Reward)
            
        traci.close()
        SimulationTime=round(timeit.default_timer()-StartTime, 1)
        
        return SimulationTime

    # Proceed with the simulation in Sumo            
    def Simulate(self, StepsToDo):
        if (self.Step+StepsToDo)>=self.MaxSteps:
            StepsToDo=self.MaxSteps-self.Step
        
        while StepsToDo>0:
            traci.simulationStep()
            self.Step+=1
            StepsToDo-=1
            QueueLength=self.GetQueueLength()
            self.EpisodeQueueLength.append(QueueLength)

    # Collect the waiting time for every car in the Incoming Roads            
    def CollectWaitingTimes(self):
        IncomingRoads=["E2TL", "N2TL", "W2TL", "S2TL"]
        CarList=traci.vehicle.getIDList()
        for CarID in CarList:
            WaitTime=traci.vehicle.getAccumulatedWaitingTime(CarID)
            RoadID=traci.vehicle.getRoadID(CarID) # Get Road ID on which the vehicle is
            if RoadID in IncomingRoads: # Considers only waiting time of cars in incoming roads
                self.WaitingTimes[CarID]=WaitTime
            else:
                if CarID in self.WaitingTimes: # Car that was tracked and has now cleared the intersection
                    del self.WaitingTimes[CarID]
        TotalWaitingTime=sum(self.WaitingTimes.values())
        
        return TotalWaitingTime

    # Picks the best action based on the current state of the environment            
    def ChooseAction(self,state):
        return np.argmax(self.Model.PredictOne(state))

    # Activates the green correct green light combination
    def SetYellowPhase(self, OldAction):
        YellowPhaseCode=OldAction*2+1
        traci.trafficlight.setPhase("TL", YellowPhaseCode)    

    # Activates the green correct green light combination
    def SetGreenPhase(self,ActionNumber):
        if ActionNumber==0:
            traci.trafficlight.setPhase("TL", NS_Green)
        elif ActionNumber==1:
            traci.trafficlight.setPhase("TL",NSL_Green)
        elif ActionNumber==2:
            traci.trafficlight.setPhase("TL",EW_Green)
        elif ActionNumber==3:
            traci.trafficlight.setPhase("TL",EWL_Green)

    # Get the number of cars with no speed in all the incoming lanes
    def GetQueueLength(self):
        HaltN=traci.edge.getLastStepHaltingNumber("N2TL")
        HaltS=traci.edge.getLastStepHaltingNumber("S2TL")
        HaltE=traci.edge.getLastStepHaltingNumber("E2TL")
        HaltW=traci.edge.getLastStepHaltingNumber("W2TL")
        
        QueueLength=HaltN+HaltS+HaltE+HaltW
        return QueueLength

    # Retrieves the state of the junction from sumo              
    def GetState(self):
        State=np.zeros(self.NumStates)
        CarList=traci.vehicle.getIDList()
        
        for CarID in CarList:
            LanePosition=traci.vehicle.getLanePosition(CarID)
            LaneID=traci.vehicle.getLaneID(CarID)
            LanePosition=750-LanePosition
            
            if LanePosition<7:
                LaneCell=0
            elif LanePosition<14:
                LaneCell=1
            elif LanePosition<21:
                LaneCell=2
            elif LanePosition<28:
                LaneCell=3
            elif LanePosition<40:
                LaneCell=4
            elif LanePosition<60:
                LaneCell=5
            elif LanePosition<100:
                LaneCell=6
            elif LanePosition<160:
                LaneCell=7
            elif LanePosition<400:
                LaneCell=8
            elif LanePosition<750:
                LaneCell=9
            
            # Any lanes ending with 3 are the "Turn Left Only" Lanes
            if LaneID == "W2TL_0" or LaneID == "W2TL_1" or LaneID == "W2TL_2":
                LaneGroup = 0
            elif LaneID == "W2TL_3":
                LaneGroup = 1
            elif LaneID == "N2TL_0" or LaneID == "N2TL_1" or LaneID == "N2TL_2":
                LaneGroup = 2
            elif LaneID == "N2TL_3":
                LaneGroup = 3
            elif LaneID == "E2TL_0" or LaneID == "E2TL_1" or LaneID == "E2TL_2":
                LaneGroup = 4
            elif LaneID == "E2TL_3":
                LaneGroup = 5
            elif LaneID == "S2TL_0" or LaneID == "S2TL_1" or LaneID == "S2TL_2":
                LaneGroup = 6
            elif LaneID == "S2TL_3":
                LaneGroup = 7
            else:
                LaneGroup = -1
                
            if LaneGroup >= 1 and LaneGroup <= 7:
                CarPosition = int(str(LaneGroup) + str(LaneCell))  # composition of the two postion ID to create a number in interval 0-79
                ValidCar = True
            elif LaneGroup == 0:
                CarPosition = LaneCell
                ValidCar = True
            else:
                ValidCar = False  # Doesnt detect cars crossing the intersection or driving away from it

            if ValidCar:
                State[CarPosition]=1
        
        return State