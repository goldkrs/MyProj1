import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import ZeroPadding2D,Convolution2D,MaxPooling2D
from tensorflow.keras.layers import Dense,Dropout,Softmax,Flatten,Activation,BatchNormalization
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import tensorflow.keras.backend as K
import os
import shutil
from sklearn.cluster import DBSCAN
import cv2
import face_recognition
from os import listdir

import pickle
import os
import datetime
import json


def load_dict(file_path):
	# Open the file for reading
	with open(file_path, "r") as fp:
	    # Load the dictionary from the file
	    loaded_dict = json.load(fp)
	return loaded_dict




SAMPLE_IMAGE_TRAINED_MAX_ID = 1
ACCURACY_THRESHOLD = .85

def load_vggface():
	#Define VGG_FACE_MODEL architecture
	model = Sequential()
	model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
	model.add(Convolution2D(64, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))
	model.add(Convolution2D(4096, (7, 7), activation='relu'))
	model.add(Dropout(0.5))
	model.add(Convolution2D(4096, (1, 1), activation='relu'))
	model.add(Dropout(0.5))
	model.add(Convolution2D(2622, (1, 1)))
	model.add(Flatten())
	model.add(Activation('softmax'))


	model.load_weights('vgg_face_weights.h5')

	vgg_face=Model(inputs=model.layers[0].input,outputs=model.layers[-2].output)

	return vgg_face



def get_cluster_encodings(person_id):
	encodings = []
	for x in db.faces.find({"customer": int(person_id)}):
		encodings.append(pickle.loads(x['cluster_encoding']))
	return encodings

def predict_id(test_image, vgg_face, classifier_model):
	test_face = cv2.resize(test_image,(224,224))
	test_face=np.expand_dims(test_face,axis=0)
	test_face=preprocess_input(test_face)
	img_encode=vgg_face(test_face)

	# Make Predictions
	embed=K.eval(img_encode)
	person=classifier_model.predict(embed)
	print(person.item(np.argmax(person)), np.argmax(person))
	if person.item(np.argmax(person)) > ACCURACY_THRESHOLD:
		return np.argmax(person)
	else:
		return -1

def load_classifier():
	# Load saved model
	if os.path.exists('face_classifier_model.h5') == True:
		classifier_model=tf.keras.models.load_model('face_classifier_model.h5')
		return classifier_model
	else:
		return None

def get_image_id(image, vgg_face, classifier_model):
	if classifier_model is not None:
		return predict_id(image, vgg_face, classifier_model)
	else:
		return -1

def get_data(image):
	boxes = (0,image.shape[1], image.shape[0],0)
	#boxes = face_recognition.face_locations(image, model='cnn')
	#print(imgfile)
	encodings = face_recognition.face_encodings(image, [boxes])
	# build a dictionary of the image path, bounding box location,
	# and facial encodings for the current image
	d = [{"encoding": enc}
		for (box, enc) in zip([boxes], encodings)]
	return d

def predict_image(image):
	f_id = get_image_id(image, vgg_face, classifier_model)
	print(f_id)
	if str(f_id) in id_name_dict:

		return id_name_dict[str(f_id)]

vgg_face = load_vggface()
classifier_model = load_classifier()
id_name_dict = load_dict('id_name_dict.json')

#imageLoaded = cv2.imread('kis.jpg', cv2.IMREAD_UNCHANGED)
#print('Prediction', predict_image(imageLoaded))



