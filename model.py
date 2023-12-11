import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # doesnt show any annoying tensorflow warnings
import numpy as np
import sys

from tensorflow import keras
from keras import layers,losses
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.models import load_model

class TrainModel:
    # Basic class definition
    def __init__(self, NumLayers, width, BatchSize, LearningRate, InputDimension, OutputDimension, model=None):
        self.InputDimension = InputDimension
        self.OutputDimension = OutputDimension
        self.BatchSize = BatchSize
        self.LearningRate = LearningRate
        if model:
            self.model = model
        else:
            self.model = self.BuildModel(NumLayers, width)
    
    #Builds a fully connected nueral network
    def BuildModel(self, NumLayers, width):
        Input=keras.Input(shape=(self.InputDimension,))
        Output=layers.Dense(width, activation='relu')(Input)
        
        ## AC
        # Output = layers.BatchNormalization()(Output)
        ## AC
        
        for _ in range(NumLayers):
            Output=layers.Dense(width, activation='relu')(Output)
        Output = layers.Flatten()(Output)
        
        
        ## AC
        # Output = layers.Reshape((-1, 400))(Output)
        # Output = layers.LSTM(units=400)(Output)
        ## AC


        ## AC
        # Output = layers.Reshape((-1, 400))(Output)
        # Output = layers.GRU(units=400)(Output)
        ## AC


        Output=layers.Dense(self.OutputDimension,activation='linear')(Output)
        Model=keras.Model(inputs=Input,outputs=Output,name='MyModel')
        Model.compile(loss=losses.mean_squared_error,optimizer=Adam(learning_rate=self.LearningRate))
        return Model
    
    # Predicts Action Value from a single state
    def PredictOne(self, state):
        state=np.reshape(state, [1, self.InputDimension])
        return self.model.predict(state)
    
    # Predicts Action Values from a Batch of States
    def PredictBatch(self, states):
        return self.model.predict(states)
    
    # Trains the Nueral Network using the updates Q-Values
    def TrainBatch(self, states, qsa):
        self.model.fit(states, qsa, epochs=1, verbose=0)
        
    # Saves the current model in the given path as a .h5 file and a model architecture graph
    def SaveModel(self, path):
        self.model.save(os.path.join(path,'TrainedModel.h5'))
        plot_model(self.model,to_file=os.path.join(path, 'ModelStructure.png'),show_shapes=True,show_layer_names=True)
    
class TestModel:
    def __init__(self, InputDimension, ModelPath):
        self.InputDimension=InputDimension
        self.Model=self.LoadModel(ModelPath)
        
    def LoadModel(self, ModelFolderPath):
        ModelFilePath=os.path.join(ModelFolderPath, 'TrainedModel.h5')
        
        if os.path.isfile(ModelFilePath):
            LoadedModel=load_model(ModelFilePath)
            return LoadedModel
        else:
            sys.exit("Model Does Not Exist")
    
    # Predicts Action Value from a single state
    def PredictOne(self, state):
        state=np.reshape(state, [1, self.InputDimension])
        return self.Model.predict(state)