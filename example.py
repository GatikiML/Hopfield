'''
Example Use Case Of Hopfield Network
'''
import hopfield

#Defining the image resolution in pixels
res = 64*64

#Initializing hopfield network with the image resolution
hf = HopfieldNetwork(resolution=res)

#List containing all urls / image path / PIL image objects to train the network
images = ["Image1", "Image2", "Image3"]

#Training the Hopfield network
hf.train(images, url=True)

#Retriving the Orignal image state
retrived_state = hf.retrieve(pattern="BlurredImageURL", url=True)

#Ploting the State into a Image
plot_state(retrived_state, pixel=64)
