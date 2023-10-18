import matplotlib.pyplot as plt
from PIL import Image
import random
import os

dataFolder = r'C:\Users\albed\Desktop\NeuralNetwork\Facial Recognition Neural Network\dataImages'

class processImage():
    def __init__(self):
        pass

    def image(self, user):
        folderDecendants = os.listdir(dataFolder + f'\{user}')

        # Finds all PNG items within a folder
        folderDecendants = [item for item in folderDecendants if item.endswith('.png')]
        return random.choice(folderDecendants)

    def randomImage(self):
        possibilities = ['albe', 'chichi', 'dad', 'mom', 'rudy']
        picked = possibilities[random.randint(0, 4)]

        directory = self.image(picked)
        imagePath = os.path.join(dataFolder, picked, directory)

        return [imagePath, picked]
        
    def convertImage(self, image, pixelSize, display):
        # converts data to greyscale
        image = Image.open(image).convert('L')
        image.thumbnail((pixelSize, pixelSize))

        width, height = image.size
        pixels = []

        for y in range(height):
            for x in range(width):
                grayscale_value = image.getpixel((x, y))
                pixels.append(grayscale_value)

        if display:
            plt.imshow(image, cmap='gray')
            plt.show()

        return pixels


# dataClass = processImage()
# randomImage = dataClass.randomImage()
# pixels = dataClass.convertImage(randomImage[0], 128, True)
#print(pixels)