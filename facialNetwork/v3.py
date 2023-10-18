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

        # tempData is what the input gets passed through
        # beforeInput is the saved list of weights before getting input
        # layers it the list of weights that gets saved (best performance)
        self.tempData = self.beforeInput = self.layers
        self.costData = [10, 0] # perm, temp

    # values each neuron in each layer with either random values 
    # ('ran') or predefined weights. Intended to be random
    # with the only exception being to load weights from past training
    def valueNeurons(self, weights, ran):
        # Loops through each layer
        for layerIndex in range(len(self.tempData)):
            # loops through each neuron in 'layerIndex'
            for neuronIndex in range(len(self.tempData[layerIndex])):
                if ran:
                    w, b = random.randint(0, 100)/100, 0
                else:
                    w, b = weights[layerIndex][neuronIndex], 0
                    
                self.tempData[layerIndex][neuronIndex] = w
        
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
        cost = 0

        # loop through each layer
        for cl in range(len(self.tempData)):
            currentLayer = self.tempData[cl]

            # checks to see if its in a hidden layer/input neuron
            if (cl + 1) != len(self.tempData):
                nextLayer = self.tempData[cl + 1]
                # loop through each neuron

                for neuron in range(len(nextLayer)):
                    # values the neuron
                    neuronValue = self.outputNeuron(currentLayer, nextLayer[neuron])
                    nextLayer[neuron] = neuronValue
            
            # final output
            else:
                # hashmap needs to be edited to repurpose the network
                # right now it represents the possible outputs 
                hm = {'albe': 0, 'dad' : 1, 'mom': 1, 'rudy': 1, 'chichi': 1}
                expectedOutput = hm[expectedOutput]
                actualOutput = currentLayer.index(max(currentLayer))

                # loops through all output
                for i in range(len(currentLayer)):
                    # if this output is expected, then subtract 1
                    if i == expectedOutput:
                        cost += (currentLayer[i] - 1) ** 2
                    else:
                        cost += currentLayer[i] ** 2
        
        # returns cost + correct/incorrect
        # by default, (actual == expected) should return a bool
        # but I'm specifying the dataType just to be safe
        return [cost, bool(actualOutput == expectedOutput)]

    # trains the network for the most optimal weights
    def trainNetwork(self, tests: int, trials: int):
        # creates starting values for neurons
        self.valueNeurons(None, True)
        self.layers = self.beforeInput = self.tempData

        # loops (tests) times to generate new
        # weights, biases, & costs
        for testNum in range(tests):
            # self.data[place] = [weight, bias, cost]
            # EX -> bias = self.data[place][1]

            startingTime = time.time()
            for trialCount in range(trials):
                randomImage = facialData.randomImage()
                pixels = facialData.convertImage(randomImage[0], self.inputSize, False)

                testInfo = self.testNetwork(pixels, randomImage[1])
                self.costData[1] += testInfo[0]
                # print(testInfo[1])

            # Compute time is only used to test optimizations
            # print(f'Compute Time: {time.time() - startingTime}')

            # average cost amongst trials
            self.costData[1] /= trials

            # if a new best cost is found
            if self.costData[1] < self.costData[0]:
                # changes the permanent data to equal the 
                # best performant 
                self.layers = self.beforeInput
                self.costData[0] = self.costData[1]

                print(Fore.GREEN + f'New Best Cost: {self.costData[0]}' + Fore.WHITE)

            else:  
                print(Fore.RED + f'Failed Cost: {self.costData[1]}' + Fore.WHITE)

            self.costData[1] = 0

            # -> Value neurons here (backpropogate/whatever)
            self.valueNeurons(None, True)
            self.beforeInput = self.tempData


randomImage = facialData.randomImage()[0]
pixels = facialData.convertImage(randomImage, 192, False)

# network ( [Inputs, Layer Xi, Outputs] )
neural = network( [len(pixels), 10, 2] )
neural.trainNetwork(50, 5)


# lines 163-172 are used souly for testing the networks accuracy
# currently, it has about 25% accuracy with randomly generated weights

accuracy = 0
for testNum in range(50):
    randomImage = facialData.randomImage()
    pixels = facialData.convertImage(randomImage[0], neural.inputSize, False)

    testInfo = neural.testNetwork(pixels, randomImage[1])
    if testInfo[1] == True:
        accuracy += 1

print(f'Network has an accuracy of {(accuracy/50) * 100}%')