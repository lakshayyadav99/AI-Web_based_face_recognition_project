from imutils import paths 
#Imutils are a series of convenience functions to make basic image processing functions such as translation, rotation, resizing, skeletonization, and displaying Matplotlib images easier with OpenCV
import face_recognition
import argparse
import pickle
#Pickle in Python is primarily used in serializing and deserializing a Python object structure. In other words, it’s the process of converting a Python object into a byte stream to store it in a file/database, maintain program state across sessions, or transport data over the network. The pickled byte stream can be used to re-create the original object hierarchy by unpickling the stream.
import cv2
import os

# construct the argument parser and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--dataset', required=True,
                    help='Path to input directory of faces + images')
parser.add_argument('-e', '--encoding', required=True,
                    help='Path to save encoded images pickle')
parser.add_argument('-d', '--detection-method', type=str, default='cnn',
                    help="Face detection model to use: 'hog' or 'cnn'")
args = vars(parser.parse_args())

# grab the paths to the input images in our dataset
print('[INFO] quantifying faces...')
imagePaths = list(paths.list_images(args['dataset']))

# initialize the list of known encoding and known names
knownEncodings = list()
knownNames = list()

# iterate over the path to each image
for (i, imagePath) in enumerate(imagePaths):
    # extract the name from path
    # for example, path: dataset/name/image1.jpg
    # if we use os.path.sep to split it
    # and pick the second from last index
    # we will get 'name' which, in this case, is a label
    print(f'[INFO] processing images {i+1}/{len(imagePaths)}')
    name = imagePath.split(os.path.sep)[-2]
    
    # load the image and convert rom BGR (OpenCV default)
    # to RGB (dlib default)
    # dlib = the dlib is used to estimate the location of 68 coordinates (x, y) that map the facial points on a person's face like image below.
    image = cv2.imread(imagePath)
    # reduce size by half for faster processing
    image = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # detect the face in each frame 
    # and return (x,y)-coordinate of the bounding box
    boxes = face_recognition.face_locations(rgb, model=args['detection_method'])
    
    # compute the face embedding
    encodings = face_recognition.face_encodings(rgb, boxes)
    
    # iterate over encodings
    # the reason we have to iterate over encoding even it's a single image
    # is sometimes one person's face might appear in more than 1 place in the image
    # for example, that person is looking at the mirror

#Face Encodings
#It is a face for us. But, for our algorithm, it is only an array of RGB values — that matches a pattern that the it has learnt from the data samples we provided to it.
#For face recognition, the algorithm notes certain important measurements on the face — like the color and size and slant of eyes, the gap between eyebrows, etc. All these put together define the face encoding — the information obtained out of the image — that is used to identify the particular face.
#To get a feel of what is read from the face, let us have a look at the encodings that we read.



    for encoding in encodings:
        # add each encoding and name to the list
        knownEncodings.append(encoding)
        knownNames.append(name)
        
# save encoding to pickle
print('[INFO] saving encodings...')
data = {'encodings': knownEncodings, 'names': knownNames}
with open(args['encoding'], 'wb') as file:
    file.write(pickle.dumps(data))
    