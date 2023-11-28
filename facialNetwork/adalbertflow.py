# Adalberto de Hombre
# 1/11/2023 (v0.0.0) - 11/27/23 (v3.1.3)
# Neural Network Class
# 'Adalbertflow' :)

from colorama import Fore
import numpy as np
import random
import math
import sys

class network():
    '''
    networkInfo = Diagram of the networks layout
        ex: [inputSize, x0,- xm, outputSize]
    
    outputLayer = Diagram of the output layout
        ex: {'item0': 0, 'item1': 1, 'neuron3': 2}
    
    trainingInputs = class representing the databank
        must have 'pullRandom' function that process data into [x0, x1,- xm]
        trainingInputs -> pullRandom():: returns [processedData, value (EX: 'item0')]
    
    trainingBias = the output that should be relayed to remove overfitting
        ex: if outputLayer looks like {'item0': 0, 'item1': 1, 'neuron3': 1}
            then a trainingBias may be set to 'item0' to intentionally
            relay it throughout training
    '''
    def __init__(self, networkInfo, outputLayer, trainingInputs, trainingBias=None):
        self.inputSize = networkInfo[0]
        self.layers = self.storedData = []
        self.outputLayer = outputLayer

        # loops through network info to create
        # all the nessesary neurons/layers
        for neurons in range(len(networkInfo)):
            # networkInfo[place] = size
            # adds a new layer inside self.layers
            place: int = neurons
            size: int = networkInfo[neurons]

            # initializes a new layer
            self.layers.append(place)
            self.layers[place] = []

            # adds neurons inside of each layer
            for layerPos in range(size):
                self.layers[place].append(layerPos)
                self.layers[place][layerPos] = 0 

        # tempData -> input gets passed through
        # beforeInput -> saved list of weights (before input)
        # layers -> list of weights that gets saved (best performant)
        self.tempData = self.beforeInput = self.layers
        self.costData = [5000, 5000] # perm, temp
        self.testingIterations = 0
        self.trainingInputs = trainingInputs
        self.trainingBias = trainingBias
        
    # values EVERY NEURON IN THE NETWORK
    def valueNeurons(self, weights=None ):
        if weights: self.tempData = weights
        else: self.tempData = [[random.random() for _ in r] for r in self.tempData]
            
    # y(i) = tanh ( bias + inpu(i) * weights(i) )
    # inpu = previous layer, weights = current layer
    def outputNeuron(self, inputs, weight, bias=0):
        xPos = sum(weight * x for x in inputs) + bias # numpy.dot(inpu, weights)
        
        output = math.tanh(xPos)
        ''' output = math.tanh(xPos)  # 75%, 2.58
            output = math.sin(xPos)   # 56%, 2.45
            output = math.cos(xPos)   # 25%, 2.45
            output = math.asinh(xPos) # 65%, 2.8
            output = math.erf(xPos)   # 40%, 5000im/s
            output = math.tan(xPos)   # 60%, 5300im/s
        '''
        
        return float(output)

    # determines the cost of the output (sample)
    def linearReg(self, y_pred, y_true, delta=1.0):
        ''' mean sqrt        ~65.3475%avg      bias 1/3   17im/s
            reLu             ~44.043%avg       bias 1/3   18im/s
            multi-binary     ~34.05%avg        bias 1/3   14im/s
            hubert/robust    ~18.42345%avg     bias 1/3   15im/s '''
        return 0.5* ((y_true[1] - y_pred[1])**2 + (y_true[0] - y_pred[0])**2)

    # tests network by going through ALL LAYERS
    # ONE CALL = full network call
    def testNetwork(self, input, expectedOutput): # , ret=None):
        self.tempData[0] = input

        # loop through each layer
        for cl in range(length:= len(self.tempData) - 1):
            if cl== 0: continue
            
            currentLayer = self.tempData[cl]
            nextLayer = self.tempData[cl + 1]

            # loop through each neuron
            for neuron in range(len(nextLayer)):
                # values the neuron
                bias = random.randint(0, 100)/100
                neuronValue = self.outputNeuron(currentLayer, nextLayer[neuron], bias)
                nextLayer[neuron] = neuronValue
        
        # calculate cost (N0 = Albe, N1 = Not Albe)
        cost, outputLayer = 0, self.tempData[-1]
        # expectedOutput = self.expectedOutput[expectedOutput]
        # expectedOutput is the NEURON that should be activated (1-#output)

        actualOutput = outputLayer.index(max(outputLayer))

        # loops through all output
        for i in range(len(outputLayer)):
            # (Expected output - actual output) ^2
            expected:bool = (i == expectedOutput)
            outputLayer[i] -= (0 if expected else 1)

            cost += outputLayer[i]**2 #math.pow( outputLayer[i],  2 )
        
        '''
        BINARY CROSS REGRESSION:;

        ypred = []
        for i in range(len(outputLayer)):
            expected:bool = (i == expectedOutput)

            ypred.append(len(ypred)) 
            ypred[len(ypred) - 1] = 1.0 if expected else 0.0

        cost = self.binaryCross(ypred, outputLayer)'''
            
        # if ret: ret[1] += cost
        return [cost, bool(actualOutput == expectedOutput), actualOutput]

    # loading bar animation in output
    def loadingBar(self, iteration, total, barLen=50):
        arrow = '=' * int(round((iteration / total) * barLen))
        spaces = ' ' * (barLen - len(arrow))
        sys.stdout.write(f'{arrow}>{spaces} {iteration/total * 100:.2f}%\r')
        sys.stdout.flush()

    # trains the network for the most optimal weights
    def trainNetwork(self, tests=25, trials=5):
        self.layers = self.beforeInput = self.tempData
        trials += trials % 2
        print('Training...')

        # loops (tests) times to generate new
        # weights, biases, & costs
        for testNum in range(tests):
            # self.data[place] = [weight, bias, cost]
            # EX -> bias = self.data[place][1]
            self.costData[1] = 0 

            for trialCount in range(trials):
                bias = None
                if self.trainingBias and trialCount == 1:
                    bias = self.trainingBias

                # randomImage[0] = Formatted inputs (EX: [0, 0, 1, 1, 0.5, 0] )
                # randomImage[1] = Expected outputs (EX: 'bus', 'stop sign', ect)
                randomImage = self.trainingInputs.pullRandom(self.inputSize, bias)
                expectedOutput = self.convertEXPO(randomImage[1])

                testInfo = self.testNetwork(randomImage[0], expectedOutput)
                self.costData[1] += testInfo[0]
                self.testingIterations += 1

                self.loadingBar(self.testingIterations, tests*trials)

            self.costData[1] /= trials
            
            # if a new best cost is found
            if bestCost:= self.costData[1] < self.costData[0]:
                self.layers = self.beforeInput # updates layers (Best Weights)
                self.costData[0] = self.costData[1] # updates lowest cost
                
            self.valueNeurons()
            self.beforeInput = self.tempData
        
        print(f'\nCost: {self.costData[0]}' + Fore.WHITE)
        self.testingIterations = 0

    # converts expectedOutput to activated neuron
    def convertEXPO(self, output):
        try: return self.outputLayer[output]
        except: return {v:k for k, v in self.outputLayer.items()}[output]

    # reads/updates stored weights from specified database
    def accessDatabase(self, read, fileLocation):
        if not read:
            condensedData = self.beforeInput
            condensedData[0] = []
            condensedData = '\n'.join(map(str, condensedData))

            with open(fileLocation, "w") as file:
                file.write(condensedData)

        else:
            with open(fileLocation, "r") as file:
               self.beforeInput = file.read().split('\n')
