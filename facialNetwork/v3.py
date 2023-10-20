# Adalberto (Albe) de Hombre
# 10/17/2023
# Facial Recognition AI

from imageSplit import processImage
from colorama import Fore
import random
import math
import time
facialData = processImage()


class network():
    def __init__(self, networkInfo):
        self.inputSize = networkInfo[0]
        self.layers = self.storedData = []

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

        # Data hashmaps
        self.expectedOutput = {'albe': 0, 'dad' : 1, 'mom': 1, 'rudy': 1, 'chichi': 1}
        self.booleanColors = {True: Fore.GREEN, False: Fore.RED} # Purely astetic

        # tempData -> input gets passed through
        # beforeInput -> saved list of weights (before input)
        # layers -> list of weights that gets saved (best performant)
        self.tempData = self.beforeInput = self.layers
        self.costData = [10, 0] # perm, temp

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

        return float(output)

    # tests network by going through ALL LAYERS
    # ONE CALL = full network call
    def testNetwork(self, input, expectedOutput):
        self.tempData[0] = input

        # loop through each layer
        for cl in range(length:= len(self.tempData) - 1):
            currentLayer = self.tempData[cl]
            nextLayer = self.tempData[cl + 1]

            # loop through each neuron
            for neuron in range(len(nextLayer)):
                # values the neuron
                neuronValue = self.outputNeuron(currentLayer, nextLayer[neuron])
                nextLayer[neuron] = neuronValue
        
        # calculate cost (N0 = Albe, N1 = Not Albe)
        cost, currentLayer = 0, self.tempData[-1]
        expectedOutput = self.expectedOutput[expectedOutput]
        actualOutput = currentLayer.index(max(currentLayer))

        # loops through all output
        for i in range(len(currentLayer)):
            # if this output is not expected, then subtract 1
            currentLayer[i] -= 1 if (i != expectedOutput) else 0
            cost += currentLayer[i] ** 2
            
        return [cost, bool(actualOutput == expectedOutput), actualOutput]

    # trains the network for the most optimal weights
    def trainNetwork(self, tests: int, trials: int):
        # creates starting values for neurons
        self.valueNeurons()
        self.layers = self.beforeInput = self.tempData

        # loops (tests) times to generate new
        # weights, biases, & costs
        for testNum in range(tests):
            # self.data[place] = [weight, bias, cost]
            # EX -> bias = self.data[place][1]

            startingTime = time.time()
            for trialCount in range(trials):
                randomImage = facialData.image()
                pixels = facialData.convertImage(randomImage[0], self.inputSize, False)

                testInfo = self.testNetwork(pixels, randomImage[1])
                self.costData[1] += testInfo[0]

            # average cost amongst trials
            self.costData[1] /= trials
            
            # if a new best cost is found
            if bestCost:= self.costData[1] < self.costData[0]:
                self.layers = self.beforeInput # updates layers (Best Weights)
                self.costData[0] = self.costData[1] # updates lowest cost
                
            print(self.booleanColors[bestCost] + f'Cost: {self.costData[1]}' + Fore.WHITE)
            print(f'--> Compute Time: {time.time() - startingTime}\n')
            self.costData[1] = 0    
                
            # -> Value neurons here (backpropogate/whatever)
            self.valueNeurons()
            self.beforeInput = self.tempData


randomImage = facialData.image()[0]
pixels = facialData.convertImage(randomImage, 192, False)

# network ( [Inputs, Layer Xi, Outputs] )
neural = network( [len(pixels), 10, 2] )
neural.trainNetwork(50, 5)


# accuracy testing (non-visual)
accuracy = 0
print(Fore.MAGENTA + f'\nTesting Accuracy... ' + Fore.WHITE)

for testNum in range(50):
    ra = facialData.image()
    pix = facialData.convertImage(ra[0], neural.inputSize, False)

    testInfo = neural.testNetwork(pix, ra[1])
    if testInfo[1] == True:
        accuracy += 1

print(f'Network has an accuracy of {(accuracy/50) * 100}%')


# Visual testing
for _ in range(4):
    if _%4!=0:
        ra = facialData.image()
    else:
        ra = facialData.image('albe')

    pix = facialData.convertImage(ra[0], neural.inputSize, True)

    testInfo = neural.testNetwork(pix, ra[1])
    formatted = 'Is' if (testInfo[2] == 0) else 'Not'
    print(f'{formatted} a picture of albe.. (Thats a {testInfo[1]} statement)')