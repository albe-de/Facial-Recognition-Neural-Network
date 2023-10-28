# Adalberto (Albe) de Hombre
# 10/17/2023
# Facial Recognition AI

# repurpose network by editing network -> 'convertExpo'

from imageSplit import processImage
from colorama import Fore
import threading
import random
import math
import time
import sys
import os
facialData = processImage()


class network():
    def __init__(self, networkInfo, outputLayer):
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
        self.costData = [5000, 0] # perm, temp
        
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
        ''' output = math.tanh(xPos) # 60%, 2.58
            output = math.sin(xPos)  # 56%, 2.45
            output = math.cos(xPos)  # 25%, 2.45
        '''
        
        return float(output)

    # tests network by going through ALL LAYERS
    # ONE CALL = full network call
    def testNetwork(self, input, expectedOutput, ret=None):
        self.tempData[0] = input

        # loop through each layer
        for cl in range(length:= len(self.tempData) - 1):
            if cl == 0: continue
            
            currentLayer = self.tempData[cl]
            nextLayer = self.tempData[cl + 1]

            # loop through each neuron
            for neuron in range(len(nextLayer)):
                # values the neuron
                neuronValue = self.outputNeuron(currentLayer, nextLayer[neuron])
                nextLayer[neuron] = neuronValue
        
        # calculate cost (N0 = Albe, N1 = Not Albe)
        cost, currentLayer = 0, self.tempData[-1]
        # expectedOutput = self.expectedOutput[expectedOutput]
        # expectedOutput is the NEURON that should be activated (1-#output)

        actualOutput = currentLayer.index(max(currentLayer))

        # loops through all output
        for i in range(len(currentLayer)):
            # (Expected output - actual output) ^2
            expected:bool = (i == expectedOutput)
            currentLayer[i] -= (0 if expected else 1)
            cost += math.pow( currentLayer[i],  2 )
            
        if ret: ret[1] += cost
        return [cost, bool(actualOutput == expectedOutput), actualOutput]

    # loading bar animation in output
    def loadingBar(self, iteration, total, barLen=50):
        arrow = '=' * int(round((iteration / total) * barLen))
        spaces = ' ' * (barLen - len(arrow))
        sys.stdout.write(f'[{arrow}>{spaces}] {(iteration / total) * 100:.2f}%\r')
        sys.stdout.flush()

    # trains the network for the most optimal weights
    def trainNetwork(self, tests: int, trials: int):
        totalIterations = 1
        os.system('cls')

        booleanColors = {True: Fore.GREEN, False: Fore.RED}
        self.layers = self.beforeInput = self.tempData
        print('Training...')

        # loops (tests) times to generate new
        # weights, biases, & costs
        for testNum in range(tests):
            # self.data[place] = [weight, bias, cost]
            # EX -> bias = self.data[place][1]

            startingTime, threads = time.time(), [None]*trials

            for trialCount in range(trials):
                randomImage = facialData.image()
                pixels = facialData.convertImage(randomImage[0], self.inputSize, False)

                # testInfo = self.testNetwork(pixels, randomImage[1])
                arguments = (pixels, self.convertEXPO(randomImage[1]), self.costData)
                threads[trialCount] = threading.Thread(target=self.testNetwork, args=arguments)
                threads[trialCount].start()

                self.loadingBar(totalIterations, tests*trials)
                totalIterations += 1

            # waits for all tests to finish and adds up all cost
            for thread in threads: thread.join()
            self.costData[1] /= trials
            
            # if a new best cost is found
            if bestCost:= self.costData[1] < self.costData[0]:
                self.layers = self.beforeInput # updates layers (Best Weights)
                self.costData[0] = self.costData[1] # updates lowest cost
                
            self.costData[1] = 0 
            self.valueNeurons()
            self.beforeInput = self.tempData
        
        print('')
        print(f'Cost: {self.costData[0]}' + Fore.WHITE)

    # converts expectedOutput to activated neuron
    def convertEXPO(self, output):
        try: return self.outputLayer[output]
        except: return {v:k for k, v in self.outputLayer.items()}[output]


randomImage = facialData.image()[0]
pixels = facialData.convertImage(randomImage, 192, False)

# network ( [Inputs, Layer Xo -Xm, outputLen] )
outputLayer = {'albe': 0, 'mom': 1, 'dad':2, 'rudy': 3, 'chichi': 4}
neural = network([len(pixels), 10, 5, len(outputLayer)], outputLayer)
neural.trainNetwork(25, 10)


# accuracy testing (non-visual)
accuracy = 0
print(f'\nTesting Accuracy... ' + Fore.WHITE)

avgComputeTime = time.time()
for testNum in range(50):
    ra = facialData.image()
    pix = facialData.convertImage(ra[0], neural.inputSize, False)

    testInfo = neural.testNetwork(pix, neural.convertEXPO(ra[1]))
    if testInfo[1] == True:
        accuracy += 1

ips = math.floor(60 / (( time.time() - avgComputeTime) / 50))
print(f'Analyze\'s {ips}im/s with {(accuracy/50) * 100}% Accuracy \n')


# Visual testing
for _ in range(4):
    img = facialData.image(None if _!=3 else 'albe')
    pix = facialData.convertImage(img[0], neural.inputSize, True)

    testInfo = neural.testNetwork(pix, neural.convertEXPO(img[1]))
    formatted = Fore.GREEN if testInfo[1] else Fore.RED
    print(formatted + f'{neural.convertEXPO(testInfo[2])}' + Fore.WHITE)