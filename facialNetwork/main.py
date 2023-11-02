from imageSplit import processImage
from adalbertflow import network
from colorama import Fore
import random
import math
import time
import os

# used for training the network
class dataBank():
    def __init__(self):
        self.imageSource = processImage()

    def pullRandom(self, size=192, of=None):
        images = self.imageSource

        randomImage = images.image(of)
        pixels = images.convertImage(randomImage[0], size)
        return [pixels, randomImage[1]]
    
images = dataBank()
os.system('cls')

# pullRandom[0] = Formatted inputs (EX: [0, 0, 1, 1, 0.5, 0] )
# pullRandom[1] = Expected outputs (EX: 'bus', 'stop sign', ect)
# network ( [Inputs, Layer Xo -Xm, outputLen] )
outputLayer = {'albe': 0, 'mom': 1, 'dad':1, 'rudy': 1, 'chichi': 1}

inputSize = len(images.pullRandom()[0])
neural = network([inputSize, 10, 10, 2], outputLayer, images)
neural.trainNetwork(50, 5)

# accuracy testing (non-visual)
accuracy = 0
print(f'\nTesting Accuracy... ' + Fore.WHITE)

avgComputeTime = time.time()
for testNum in range(50):
    ra = images.pullRandom()
    testInfo = neural.testNetwork(ra[0], neural.convertEXPO(ra[1]))
    
    if testInfo[1]: accuracy += 1

ips = math.floor(60 / (( time.time() - avgComputeTime) / 50))
print(f'Analyzes {ips} im/s with {(accuracy/50) * 100:.2f}% Accuracy \n')

# 'Visual' testing
for _ in range(6):
    img = images.pullRandom(inputSize, 'albe' if _%2==0 else None)
    testInfo = neural.testNetwork(img[0], neural.convertEXPO(img[1]))

    formatted = Fore.GREEN if testInfo[1] == True else Fore.RED
    print(formatted + f'{neural.convertEXPO(testInfo[2])}' + Fore.WHITE)