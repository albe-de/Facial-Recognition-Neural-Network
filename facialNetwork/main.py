from imageSplit import processImage
from adalbertflow import network
from colorama import Fore
import threading
import math
import time
import os

dataBase = rf'E:\AA_ Windows Recovery Backup\Code Backup - Facial Rec\Facial-Recognition-Neural-Network\facialNetwork\savedWeights.txt'

# used for training the network
class dataBank():
    def __init__(self):
        self.imageSource = processImage()

    def pullRandom(self, size=192, of=None):
        images = self.imageSource

        randomImage = images.image(of)
        pixels = images.convertImage(randomImage[0], size)
        return [pixels, randomImage[1]]
    
os.system('cls')
images = dataBank()
inputSize = len(images.pullRandom()[0])

# network ( [Inputs, Layer Xo -Xm, outputLen] )
# networkInfo, outputLayer, trainingInputs
outputLayer = {'albe': 0, 'mom': 1, 'dad':1, 'rudy': 1, 'chichi': 1}
neural = network( [inputSize, 10, 10, 2], outputLayer, images, 'albe' )

# neural.trainNetwork(25, 10)
# neural.layers = neural.accessDatabase(True, dataBase)
neural.decompileNetwork(5e-5, 5e-3)
neural.accessDatabase(False, dataBase)

# # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # DEEP ACCURACY TESTING # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # #

avgComputeTime, tests = time.time(), 75
accuracy = 0

def computeAccuracy():
    global accuracy

    for testNum in range(tests):
        ra = images.pullRandom(inputSize)
        testInfo = neural.testNetwork(ra[0], outputLayer[ra[1]])

        formatted = Fore.GREEN if testInfo[1] == True else Fore.RED
        print(formatted + f'{ra[1]}' + Fore.WHITE)
        if testInfo[1]: accuracy += 1

    
print(f'\nTesting Accuracy... ' + Fore.WHITE)
threads, epochs = [], 1

for i in range(epochs):
    threads.append(i)
    threads[i] = threading.Thread(target=computeAccuracy)

for t in threads: t.start()
for t in threads: t.join()

ips = math.floor((tests * epochs) / (time.time() - avgComputeTime))
accuracy = f"{(accuracy / tests) * (100/epochs):.2f}"
print(f"Analysis {ips}/images sec at {accuracy}%")