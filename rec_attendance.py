#! /usr/bin/python

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import imutils
import pickle
import time
import cv2

# Importing required modules 
import csv 
from datetime import datetime 

run_video = True
def stop_capture():
	global run_video
	run_video = False
def rec_entry(usn,cur_date_time):
		
	# Opening the CSV file in read and 
	# write mode using the open() module 
	with open(r'entry_log.csv', 'a', newline='') as file:  
		file_write = csv.writer(file) 

		file_write.writerow([usn,cur_date_time]) 
		file.flush()
		file.close()

def capture_attendance():
	#Initialize 'currentname' to trigger only when a new person is identified.
	currentname = "unknown"
	#Determine faces from encodings.pickle file model created from train_model.py
	encodingsP = "encodings.pickle"

	# load the known faces and embeddings along with OpenCV's Haar
	# cascade for face detection
	print("[INFO] loading encodings + face detector...")
	data = pickle.loads(open(encodingsP, "rb").read())

	# initialize the video stream and allow the camera sensor to warm up
	# Set the ser to the followng
	# src = 0 : for the build in single web cam, could be your laptop webcam
	# src = 2 : I had to set it to 2 inorder to use the USB webcam attached to my laptop
	vs = VideoStream(src=0,framerate=10).start()
	#vs = VideoStream(usePiCamera=True).start()
	time.sleep(2.0)

	# start the FPS counter
	fps = FPS().start()

	global run_video

	# loop over frames from the video file stream
	while run_video:
		# grab the frame from the threaded video stream and resize it
		# to 500px (to speedup processing)
		frame = vs.read()
		if frame is not None:
			frame = imutils.resize(frame, width=500)
			# Detect the fce boxes
			boxes = face_recognition.face_locations(frame)
			# compute the facial embeddings for each face bounding box
			encodings = face_recognition.face_encodings(frame, boxes)
			names = []

			# loop over the facial embeddings
			for encoding in encodings:
				# attempt to match each face in the input image to our known
				# encodings
				matches = face_recognition.compare_faces(data["encodings"],
					encoding)
				name = "Unknown" #if face is not recognized, then print Unknown

				# check to see if we have found a match
				if True in matches:
					# find the indexes of all matched faces then initialize a
					# dictionary to count the total number of times each face
					# was matched
					matchedIdxs = [i for (i, b) in enumerate(matches) if b]
					counts = {}

					# loop over the matched indexes and maintain a count for
					# each recognized face face
					for i in matchedIdxs:
						name = data["names"][i]
						counts[name] = counts.get(name, 0) + 1

					# determine the recognized face with the largest number
					# of votes (note: in the event of an unlikely tie Python
					# will select first entry in the dictionary)
					name = max(counts, key=counts.get)

					rec_entry(name,datetime.now())
					print(name)
					time.sleep(10)

		

		# update the FPS counter
		fps.update()

	# stop the timer and display FPS information
	fps.stop()
	print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

	# do a bit of cleanup
	cv2.destroyAllWindows()
	vs.stop()
	run_video = True
