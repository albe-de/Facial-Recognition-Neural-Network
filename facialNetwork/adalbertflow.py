# Adalberto de Hombre
# 1/11/2023
# Neural Network Class
# 'Adalbertflow' :)

from colorama import Fore
import random
import math
import sys

class network():
    def __init__(self, networkInfo, outputLayer, trainingInputs):
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
        
    # values EVERY NEURON IN THE NETWORK
    def valueNeurons(self, weights=None):
        for Li in range(len(self.tempData)):
            for Ni in range(len(self.tempData[Li])):

                # assigns random value or predefined value to neuron
                val = random.randint(0, 1000) / 1000
                if (weights): val = weights[Li][Ni]
                    
                self.tempData[Li][Ni] = val
        
        # not all calls require an intake
        return self.tempData
            
    # y(i) = tanh ( bias + inpu(i) * weights(i) )
    # inpu = previous layer, weights = current layer
    def outputNeuron(self, inputs, weight, bias=0):
        xPos = sum(weight * x for x in inputs) + bias # numpy.dot(inpu, weights)
        
        output = math.tanh(xPos)
        ''' output = math.tanh(xPos)  # 60%, 2.58
            output = math.sin(xPos)   # 56%, 2.45
            output = math.cos(xPos)   # 25%, 2.45
            output = math.fabs(xPos)  # 78%, 2.56 (guessed notAlbe every time)
            output = math.asinh(xPos) # 65%, 2.8
        '''
        
        return float(output)

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

            cost += math.pow( outputLayer[i],  2 )
            
        # if ret: ret[1] += cost
        return [cost, bool(actualOutput == expectedOutput), actualOutput]

    # loading bar animation in output
    def loadingBar(self, iteration, total, barLen=50):
        arrow = '=' * int(round((iteration / total) * barLen))
        spaces = ' ' * (barLen - len(arrow))
        sys.stdout.write(f'{arrow}>{spaces} {iteration/total * 100:.2f}%\r')
        sys.stdout.flush()

    # trains the network for the most optimal weights
    def trainNetwork(self, tests: int, trials: int):
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
                # randomImage[0] = Formatted inputs (EX: [0, 0, 1, 1, 0.5, 0] )
                # randomImage[1] = Expected outputs (EX: 'bus', 'stop sign', ect)
                randomImage = self.trainingInputs.pullRandom(self.inputSize)
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


# goonery: 
#   * outputNeuron function
#   * make sure covertEXPO works
#   * hill stepping
#   * mess with nonlinearity
'''
randomImage = facialData.image()[0]
pixels = facialData.convertImage(randomImage, 192)

# network ( [Inputs, Layer Xo -Xm, outputLen] )
outputLayer = {'albe': 0, 'mom': 1, 'dad':1, 'rudy': 1, 'chichi': 1}
neural = network([len(pixels), 10, 5, 2], outputLayer)
neural.trainNetwork(25, 10)

///////////

# accuracy testing (non-visual)
accuracy = 0
print(f'\nTesting Accuracy... ' + Fore.WHITE)

avgComputeTime = time.time()
for testNum in range(50):
    ra = facialData.image()
    pix = facialData.convertImage(ra[0], neural.inputSize)

    testInfo = neural.testNetwork(pix, neural.convertEXPO(ra[1]))
    if testInfo[1]: accuracy += 1

ips = math.floor(60 / (( time.time() - avgComputeTime) / 50))
print(f'Analyzes {ips} im/s with {(accuracy/50) * 100:.2f}% Accuracy \n')
'''