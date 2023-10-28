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
        self.costData = [100, 0] # perm, temp
         
    # y(i) = tanh ( bias + inpu(i) * weights(i) )
    # inpu = previous layer, weights = current layer
    def A(self, inputs, weight, bias=0):
        xPos = sum(weight * x for x in inputs) + bias # numpy.dot(inpu, weights)
        output = math.tanh(xPos)

        return float(output)

    # determines cost of current weights
    # returns [cost, right/wrong, output]
    def cost(self, input, expectedOutput):
        self.tempData[0] = input

        # loop through each layer
        for cl in range(length:= len(self.tempData) - 1):
            currentLayer = self.tempData[cl]
            nextLayer = self.tempData[cl + 1]

            # loop through each neuron
            for neuron in range(len(nextLayer)):
                # values the neuron
                neuronValue = self.A(currentLayer, nextLayer[neuron])
                nextLayer[neuron] = neuronValue
        
        # calculate cost (N0 = Albe, N1 = Not Albe)
        cost, currentLayer = 0.000, self.tempData[-1]
        expectedOutput = self.expectedOutput[expectedOutput]
        actualOutput = currentLayer.index(max(currentLayer))

        # loops through all output
        for i in range(len(currentLayer)):
            # ( Actual output - Expected output ) ^2
            expected:bool = (i == expectedOutput)
            currentLayer[i] -= (0 if expected else 1)
            cost += math.pow( currentLayer[i],  2 )
            
        return [cost, bool(actualOutput == expectedOutput), actualOutput]

    # trains the network for the most optimal weights
    def backpropagate(self, tests: int, trials: int):
        for Li in range(len(self.tempData)):
            # self.tempData = self.beforeInput = self.layers
            for neuron in range(len(self.tempData[Li])):
                # steps from -50 to 50 to find best C

                bestCost = currentC = 100.000
                i = -5
                while (i < 5):
                    self.beforeInput[neuron] = self.tempData[neuron] = i

                    # runs cost on current weight [0, ..neuron=i.., 0]
                    startingTime = time.time()
                    for trialCount in range(trials):
                        randomImage = facialData.image()
                        pixels = facialData.convertImage(randomImage[0], self.inputSize, False)

                        testInfo = self.cost(pixels, randomImage[1])
                        currentC += testInfo[0]

                    # if its the lowest cost thus far, set weight to i
                    if ((currentCost := currentC / trials) < bestCost):
                        self.layers[neuron] = i
                        bestCost = currentCost

                    # prints data from this run
                    print(self.booleanColors[currentCost < bestCost] + 
                          f'Cost: {currentC} -> Weight: {i}' + Fore.WHITE)
                    print(f'--> Compute Time: {time.time() - startingTime}\n')
                    self.beforeInput[neuron] = currentC = 0
                    
                    i += 1

                print(f'Finished Layer {Li}, Neuron {neuron}: {bestCost}')
                bestCost = 100

randomImage = facialData.image()[0]
pixels = facialData.convertImage(randomImage, 192, False)

# network ( [Inputs, Layer Xi, Outputs] )
neural = network( [len(pixels), 10, 2] )
neural.backpropagate(50, 5)


# accuracy testing (non-visual)
accuracy = 0
print(Fore.MAGENTA + f'\nTesting Accuracy... ' + Fore.WHITE)

for testNum in range(50):
    ra = facialData.image()
    pix = facialData.convertImage(ra[0], neural.inputSize, False)

    testInfo = neural.cost(pix, ra[1])
    if testInfo[1] == True:
        accuracy += 1

print(f'Network has an accuracy of {(accuracy/50) * 100}%')


# Visual testing
for _ in range(4):
    img = facialData.image(None if _!=4 else 'albe')
    pix = facialData.convertImage(img[0], neural.inputSize, True)

    testInfo = neural.cost(pix, img[1])
    formatted = 'Is' if (testInfo[2] == 0) else 'Not'
    print(f'{formatted} a picture of albe.. (Thats a {testInfo[1]} statement)')