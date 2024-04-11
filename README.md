# Facial-Recognition-Neural-Network

**Useage**
This Network's (current) purpose is to determine if the photo its
analysing is a photo of me, or a photo of one of my family members.
It can be easlily repurposed to anylize live video and update in real time.

**Repurpose**
To repurpose it, youll need to make some slight edits to the inputs, 
number of layers, size of layers, and code (like the hashmap in network->testNetwork)

**License**
You can use the code for whatever you'd like so long as you dont steal/take 
credit for it and you dont sell it.
(MIT license) 
https://s3.ezgif.com/tmp/ezgif-3-bfa1128832.gif

<pre>
File Locations

backup: expired code archive (basically a save-file)
  v3.py           neural network class + execution 
  customIter.py   no use (as of now)
  hillStep.py     expired WIP of backpropogated network

dataImages (REMOVED FOR PRIVACY): data set with images of me and my family members- used to train the AI
  albe    PNG list with images of me
  chichi  PNG list with images of chichi
  dad     PNG list with images of my dad
  mom     PNG list with images of my mom
  rudy    PNG list with images of rudy
      
facialNetwork: code that the AI uses to function
  __pycache__       regular pycache file
  imageSplit.py     class to pull and process images from dataImages
  adalbertflow.py   neural network class 
  main.py           executes the main network class
    
LICENSE.txt   programs copyright license 
README.md     information about the program (where you are now :) )
todo          list of features for me to add/instructions for myself

v1.00<pre>
