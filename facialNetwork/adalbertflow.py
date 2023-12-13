# Adalberto de Hombre
# 1/11/2023 (v0.0.0) - 11/27/23 (v3.1.3)
# Neural Network Class
# 'Adalbertflow' :)

from colorama import Fore
import numpy as np
import threading
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
        self.trainingInputs = trainingInputs
        self.trainingBias = trainingBias
        
    # values EVERY NEURON IN THE NETWORK
    def valueNeurons(self, weights=None ):
        if weights: self.tempData = weights
        else: 
            self.tempData = [[random.randint(-1e4, 1e4)/1e4 for _ in r] for r in self.tempData]
            #self.tempData = [[self.outputNeuron([random.randint(-1e4, 1e4)/1e4] * 
            #random.randint(1, len(self.layers[2])), random.randint(-1e4, 1e4)/1e4) for _ in r] for r in self.tempData]

    # y(i) = tanh ( bias + inpu(i) * weights(i) )
    # inpu = previous layer, weights = current layer
    def outputNeuron(self, inputs, weight, bias=0):
        # sigmoid(dot) for nonlinearities
        xPos = sum(weight * x for x in inputs) + bias # np.dot(inpu, weights)
        output = 1/(1 + math.pow(math.e, -1*xPos))
        # output = (math.e**(2 * xPos) -1) / (math.e**(2 * xPos) +1) # -> tanh

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
                bias = random.randint(-1000, 1000)/1000
                neuronValue = self.outputNeuron(currentLayer, nextLayer[neuron], bias)
                nextLayer[neuron] = neuronValue
        
        # calculate cost (N0 = Albe, N1 = Not Albe)
        cost, outputLayer = 0, self.tempData[-1]
        # expectedOutput = self.expectedOutput[expectedOutput]
        # expectedOutput is the NEURON that should be activated (1-#output)

        try: 
            actualOutput = outputLayer.index(max(outputLayer))
        except:
            outputIdx = np.argmax(outputLayer)
            actualOutput = outputLayer[outputIdx]

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

                # training bias of 1/3
                if self.trainingBias and trialCount < (trials/3):
                    bias = self.trainingBias

                # randomImage[0] = Formatted inputs (EX: [0, 0, 1, 1, 0.5, 0] )
                # randomImage[1] = Expected outputs (EX: 'bus', 'stop sign', ect)
                randomImage = self.trainingInputs.pullRandom(self.inputSize, bias)
                expectedOutput = self.outputLayer[randomImage[1]]

                testInfo = self.testNetwork(randomImage[0], expectedOutput)
                self.costData[1] += testInfo[0]

            self.costData[1] /= trials
            
            # if a new best cost is found
            if bestCost:= self.costData[1] < self.costData[0]:
                self.layers = self.beforeInput # updates layers (Best Weights)
                self.costData[0] = self.costData[1] # updates lowest cost
                
            self.valueNeurons()
            self.beforeInput = self.tempData
        
        print(f'\nCost: {self.costData[0]}' + Fore.WHITE)
        self.testingIterations = 0

    def decompileNetwork(self, learningRate=0.01, epochs=1000):
        self.beforeInput = self.tempData
        threads = []

        # creates threads/layer for gradient decent
        for n in range(len(self.tempData)):
            if n == 0 or n+1 == len(self.tempData): continue
            threads.append(n-1)
            threads[n - 1] = threading.Thread(target=self.gradientDecent, args=(learningRate, epochs, n))

        for t in threads: t.start()
        for t in threads: t.join()

    def gradientDecent(self, learningRate, epochs, n):
        # layer is a memory address of tempData[n]
        layer = self.tempData[n]
        self.tempData[n] = np.array(layer, dtype=object)
        bestCost = [100, []]

        # loops 'epochs' times from XOffset -> (XOffset + learningrate(epochs))
        epochs = int(epochs / learningRate)
        for i in range(epochs):
            # converts layer to numpy
            gradient = np.zeros_like(self.tempData[n], dtype=object)
            self.tempData[n].tolist()
            cost = 0

            # determines cost of layer[n]'s gradient
            for trialCount in range(5):
                randomImage = self.trainingInputs.pullRandom(self.inputSize)
                expectedOutput = self.outputLayer[randomImage[1]]
                
                testInfo = self.testNetwork(randomImage[0], expectedOutput)
                cost += testInfo[0]

            if (cost / 5) < bestCost[0]:
                print(cost/5)
                bestCost = [(cost / 5), self.tempData[n]]
                self.layers[n] = self.tempData[n]
                self.beforeInput[n] = self.tempData[n]

            # shifts model by -learningRate
            self.tempData[n] = np.array(layer, dtype=object)
            self.tempData[n] -= learningRate * gradient

    # reads/updates stored weights from specified database
    def accessDatabase(self, read, fileLocation):
        if not read:
            condensedData = self.layers
            condensedData[0] = []
            condensedData = '\n'.join(map(str, condensedData))

            with open(fileLocation, "w") as file:
                file.write(condensedData)

        else:
            with open(fileLocation, "r") as file:
               self.layers = file.read().split('\n')

'''

1) Create array-collapse function for simple conversion
[[x0, x1,- xm], [x0, x1,- xm], xm] -> [x0, x1,- xm]

2) use collsapsed arrays for backpropogation
hillstepping X
gradient desent X
linear regression X

potentially use numpy arrays for a gradient ^^

'''