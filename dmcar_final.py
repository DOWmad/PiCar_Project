# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
from picar import back_wheels, front_wheels
import picar
from Line import Line
from lane_detection import color_frame_pipeline
from lane_detection import PID
import math
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the output video clip, e.g., -v out_video.mp4")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())

# initialize video writer
writer = None

# define the paths to the Stop/Non-Stop Keras deep learning model
MODEL_PATH = "signs_model.model"

# to hide warning message for tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# initialize the total number of frames that *consecutively* contain
# stop sign along with threshold required to trigger the sign alarm
TOTAL_CONSEC = 0
TOTAL_THRESH = 10		# fast speed-> low, slow speed -> high
STOP_SEC = 0
TOTAL_CONSEC_S = 0
TOTAL_CONSEC_Y = 0

# initialize is the sign alarm has been triggered
STOP = False

# load the trained CNN model
print("[INFO] loading model...")
model = load_model(MODEL_PATH)

# Grab the reference to the webcam
vs = VideoStream(src=0).start()

# detect lane based on the last # of frames
frame_buffer = deque(maxlen=args["buffer"])

# allow the camera or video file to warm up
time.sleep(2.0)

picar.setup()
db_file = "/home/pi/Desktop/dmcar-student/picar/config"
fw = front_wheels.Front_Wheels(debug=False, db=db_file)
bw = back_wheels.Back_Wheels(debug=False, db=db_file)

bw.ready()
fw.ready()

SIG = False
SPEED = 30
ANGLE = 115		# steering wheel angle: 90 -> straight
MAX_ANGLE = 20			# Maximum angle to turn right at one time
MIN_ANGLE = -20		# Maximum angle to turn left at one time
isMoving = False		# True: car is moving
posError = []			# difference between middle and car position
bw.speed = SPEED		# car speed
fw.turn(ANGLE)
TOTAL_THRESH_Y = 5
TOTAL_THRESH_H = 5
TOTAL_CONSEC_S = 0
consec_sig = 3
TOTAL_CONSEC_H = 0
TOTAL_CONSEC_R = 0
HIGH=False
# keep looping
while True:
	bw.forward()
	isMoving = True
	# grab the current frame
	frame = vs.read()

	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
	if frame is None:
		break

	# resize the frame, blur it, and convert it to the HSV
	# color space
	frame = imutils.resize(frame, width=320)
	(h, w) = frame.shape[:2]
	r = 320 / float(w)
	dim = (320, int(h * r))
	frame = cv2.resize(frame, dim, cv2.INTER_AREA)
	# resize to 320 x 180 for wide frame
	frame = frame[0:180, 0:320]
	frame = cv2.rectangle(frame,(265,105),(305,55),(0,255,0),2)
	# crop for CNN model, i.e., traffic sign location
	# can be adjusted based on camera angle
	image = frame[55:105, 265:305]


	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	#frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
	frame_buffer.append(frame)
	blend_frame, lane_lines = color_frame_pipeline(frames=frame_buffer, \
					   solid_lines=True, \
					   temporal_smoothing=True)

	# prepare the image to be classified by our deep learning network
	image = cv2.resize(image , (28, 28),cv2.INTER_AREA)
	#cv2.imshow("Image",image)
	image = image.astype("float") / 255.0
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)

	# classify the input image and initialize the label and
	# probability of the prediction

	# classify the input image
	low_speed, RR, signal, high_speed, stop, yield1 = model.predict(image)[0]
	label = "Not Stop"
	proba = 1

	soft_sum = np.exp(low_speed) + np.exp(RR) + np.exp(signal) + np.exp(high_speed) + np.exp(stop) + np.exp(yield1)

	#if (low_speed > 40 or RR > 40 or signal> 40 or high_speed > 40 or stop>40 oryield1 > 40)
	# check to see if stop sign was detected using our convolutional
	if max (low_speed,RR,signal, high_speed,stop,yield1) == low_speed:
		label = "Low"
		proba = np.exp(low_speed) / soft_sum
		label = "{}: {:.2f}%".format(label, proba * 100)
		print(label)
		print ("Low")
		bw.speed(10)
		time.sleep(4)
		bw.speed(30)

	elif max (low_speed,RR,signal, high_speed,stop,yield1) == RR:
		label = "RR"
		proba = np.exp(RR) / soft_sum
		label = "{}: {:.2f}%".format(label, proba * 100)
		print(label)
		TOTAL_CONSEC_R +=1
		if not RR and TOTAL_CONSEC_R >= 5:
			label = "RR"
			proba = np.exp(RR) / soft_sum
			bw.speed = 0
			isMoving=False
			bw.stop()
			time.sleep(2)
			bw.speed = 30
			STOP = False
			isMoving=True
			bw.forward()
			TOTAL_CONSEC_R=0
		#print ("RR")

	elif max (low_speed,RR,signal, high_speed,stop,yield1) == signal and signal >=14: #and signal >=15:
		label = "Signal"
		proba = np.exp(signal) / soft_sum
		TOTAL_CONSEC_S +=1
		label = "{}: {:.2f}%".format(label, proba * 100)
		print("SIGNAL")
		if not SIG and TOTAL_CONSEC_S >= 3: #TOTAL_THRESH:
			label = "Signal"
			SIG = True
			proba = np.exp(stop) / soft_sum
			label = "{}: {:.2f}%".format(label, proba * 100)
			#Stop the car for 5 secs
			bw.speed = 0
			isMoving=False
			bw.stop()
			time.sleep(5)
			bw.speed = 20
			SIG = False
			isMoving=True
			bw.forward()
			TOTAL_CONSEC_S=0
		print(label)
		print ("Signal")

	elif max (low_speed,RR,signal, high_speed,stop,yield1) == high_speed:
		label = "High"
		proba = np.exp(high_speed) / soft_sum
		label = "{}: {:.2f}%".format(label, proba * 100)
		#print(label)
		#print ("High")
		TOTAL_CONSEC_H+=1
		if not HIGH and TOTAL_CONSEC_H >4:
			label="High"
			HIGH = True
			bw.speed=50
			time.sleep(2)
			label = "{}: {:.2f}%".format(label, proba * 100)
			bw.speed=30
			TOTAL_CONSEC_H = 0
			HIGH = False
	elif max (low_speed,RR,signal, high_speed,stop,yield1) == stop:
		print(TOTAL_CONSEC)
		TOTAL_CONSEC +=1
		label = "Stop"
		print(label)
		proba = np.exp(stop) / soft_sum
		if not STOP and TOTAL_CONSEC >= 8:  #TOTAL_THRESH:
			label = "Stop"
			STOP = True
			proba = np.exp(stop) / soft_sum
			label = "{}: {:.2f}%".format(label, proba * 100)
			#Stop the car for 5 secs
			bw.speed = 0
			isMoving=False
			bw.stop()
			time.sleep(5)
			bw.speed = 20
			STOP = False
			isMoving=True
			bw.forward()
			TOTAL_CONSEC=0

		#After 3 secs move the car forward

	elif max (low_speed,RR,signal, high_speed,stop,yield1) == yield1:
		label = "Yield"
		TOTAL_CONSEC_Y +=1
		proba = np.exp(yield1) / soft_sum
		print(label)
		label = "{}: {:.2f}%".format(label, proba * 100)
		if not yield1 and TOTAL_CONSEC >= TOTAL_THRESH_Y:
			label = "Yield"
			proba = np.exp(yield1) / soft_sum
			bw.speed = 0
			isMoving=False
			bw.stop()
			time.sleep(2)
			bw.speed = 30
			STOP = False
			isMoving=True
			bw.forward()
			TOTAL_CONSEC_Y=0
		#print(label)
		#print ("Yield")
	
		 
	else:
		TOTAL_CONSEC=0
		STOP = False
		print("Nothing detected")

	# build the label and draw it on the frame
	#label = "{}: {:.2f}%".format(label, proba * 100)
	blend_frame = cv2.cvtColor(blend_frame, cv2.COLOR_RGB2BGR)
	blend_frame = cv2.putText(blend_frame, label, (10, 25),
		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
	cv2.imshow('blend', blend_frame)

	#if(len(lane_lines) == 2 and lane_lines[0].slope != 0):
	#for mid-position of lanes
	if(lane_lines[0].slope == 0 and lane_lines[1].slope ==0):
		isMoving = False
		bw.speed=0
		bw.stop()
		break
	else:
		y2L = h - 1
		x2L = int((y2L - lane_lines[0].bias) / (lane_lines[0].slope + .0000001))
		#print("X2L: ",x2L)
		y2R = h - 1
		x2R = int((y2R - lane_lines[1].bias) / (lane_lines[1].slope+.000001))
		#print("x2R: ",x2R)
		mid_position_lane = ( x2R + x2L ) / 2
	#else:
	#	y2L = h - 1
	#	x2L = 0
	#	y2R = h - 1
	#	x2R = int((y2R - lane_lines[1].bias) / lane_lines[1].slope)
	#	mid_position_lane = ( x2R + x2L ) / 2

		SPEED = 40

	if isMoving:
		# negative -> + ANGLE, positive -> - ANGLE
		car_position_err = w/2 - mid_position_lane
		car_position_time = time.time()
		posError.append([car_position_err, car_position_time])

		# Control Car
		# Adjust P(KP), I(KI), D(KD) values as well as portion
		# angle = PID(posError, KP=0.8, KI=0.05, KD=0.1) * 0.2
		angle = PID(posError, KP=0.8, KI=0.1, KD=0.1) * 0.25
		#print("angle: ", angle)
		#print("W", w)
		#print("Mid Pos: ",mid_position_lane)
		
		"""
		# MAX + - 20 degree
		if angle > MAX_ANGLE:
			print("angle GT max")
			angle = MAX_ANGLE
		elif angle < MIN_ANGLE:
			print("angle LT min")
			angle = MIN_ANGLE
		else:
			print("between")
		#ANGLE correction_angle
		"""
		if(angle > 5):
			ANGLE = 115 - angle
			#print ("angle gt 10")
			#print("turning left")
			fw.turn(ANGLE)
			"""
			if(ANGLE < 115):
			urn	fw.turn(125)
			elif(ANGLE > 115):
				fw.turn(105)
			"""
		elif (angle < -5):
			ANGLE = 115 - angle
			#print ("turning right")
			fw.turn(ANGLE)
		
		
		else:
			fw.turn(115)
			#print ("going straight")
		
		#print("Turning: ")
		#print("ANGLE",ANGLE)
		#print("error",car_position_err)
		# Right turn max 135, Left turn max 45
		#if ANGLE >= 135:
		#	ANGLE = 135
		#elif ANGLE <135
		#	ANGLE = 45
		#if angle < 0:
		#	fw.turn(145)
		#elif(angle >0):
		#	fw.turn(115)
		#fw.turn(125)

	# Video Writing
	if writer is None:
		if args.get("video", False):
			writer = cv2.VideoWriter(args["video"],
				0x00000021,
				15.0, (320,180), True)

	# if a video path is provided, write a video clip
	if args.get("video", False):
		writer.write(blend_frame)

	keyin = cv2.waitKey(1) & 0xFF
	keycmd = chr(keyin)

	# if the 'q' key is pressed, end program
	# if the 'w' key is pressed, moving forward
	# if the 'x' key is pressed, moving backword
	# if the 'a' key is pressed, turn left
	# if the 'd' key is pressed, turn right
	# if the 's' key is pressed, straight
	# if the 'z' key is pressed, stop a car
	if keycmd == 'q':
		bw.stop()
		break
	elif keycmd == 'w':
		isMoving = True
		bw.speed = SPEED
		bw.forward()
	elif keycmd == 'x':
		bw.speed = SPEED
		bw.backward()
	elif keycmd == 'a':
		ANGLE -= 5
		if ANGLE <= 45:
			ANGLE = 45
		#fw.turn_left()
		fw.turn(ANGLE)
	elif keycmd == 'd':
		ANGLE += 5
		if ANGLE >= 135:
			ANGLE = 135
		#fw.turn_right()
		fw.turn(ANGLE)
	elif keycmd == 's':
		ANGLE = 135
		#fw.turn_straight()
		fw.turn(ANGLE)
	elif keycmd == 'z':
		isMoving = False
		bw.stop()

# if we are not using a video file, stop the camera video stream
writer.release()
vs.stop()

# close all windows
cv2.destroyAllWindows()
