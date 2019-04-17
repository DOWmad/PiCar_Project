# USAGE
# python test_network.py --model stop_not_stop.model --image images/examples/stop001.jpg

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from imutils.video import VideoStream
from threading import Thread
import numpy as np
import imutils
import time
import cv2
import os

# construct the argument parse and parse the arguments
"""
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())
"""

#image1 = "C:/Users/ddabl/Documents/computer/DM/assignments/exer labs/2/DM-Exercise-Lab-2019/Ch5/images/stop/00000007.jpg"

MODEL_PATH = "signs_model.model"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# initialize the total number of frames that *consecutively* contain
# stop sign along with threshold required to trigger the sign alarm
TOTAL_CONSEC = 0
TOTAL_THRESH = 20

# initialize is the sign alarm has been triggered
LOW = False
RR = False
SIGNAL = False
SPEED = False
STOP = False
YIELD1 = False

# load the image
#image = cv2.imread(args["image"])
#image = cv2.imread(image1)
print("[INFO] loading model...")
model = load_model(MODEL_PATH)
#dd add
#model1 = "C:/Users/ddabl/Documents/computer/DM/assignments/auton/saved_modelv1.model"

#orig = image.copy()
# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)


# load the trained convolutional neural network
print("[INFO] loading network...")

#model = load_model(args["model"])


while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 320 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=320)

	# prepare the image to be classified by our deep learning network
	image = frame[60:120, 240:320]
	image = cv2.resize(image , (28, 28))
	image = image.astype("float") / 255.0
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)

	# classify the input image and initialize the label and
	# probability of the prediction
	
	# classify the input image
	low_speed, RR, signal, high_speed, stop, yield1 = model.predict(image)[0]

	print("low speed = ",low_speed)
	print("RR = ",RR)
	print("signal = ",signal)
	print("high speed = ",high_speed)
	print("stop = ",stop)
	print("yield = ",yield1)

	soft_sum = np.exp(low_speed) + np.exp(RR) + np.exp(signal) + np.exp(high_speed) + np.exp(stop) + np.exp(yield1) 

	if max (low_speed,RR,signal, high_speed,stop,yield1) == low_speed:
		label = "Low"
		proba = np.exp(low_speed) / soft_sum
		label = "{}: {:.2f}%".format(label, proba * 100)
		print(label)
		print ("Low")

	elif max (low_speed,RR,signal, high_speed,stop,yield1) == RR:
		label = "RR"
		proba = np.exp(RR) / soft_sum
		label = "{}: {:.2f}%".format(label, proba * 100)
		print(label)
		print ("RR")

	elif max (low_speed,RR,signal, high_speed,stop,yield1) == signal:
		label = "Signal"
		proba = np.exp(signal) / soft_sum
		label = "{}: {:.2f}%".format(label, proba * 100)
		print(label)
		print ("Signal")

	elif max (low_speed,RR,signal, high_speed,stop,yield1) == high_speed:
		label = "High"
		proba = np.exp(high_speed) / soft_sum
		label = "{}: {:.2f}%".format(label, proba * 100)
		print(label)
		print ("High")

	elif max (low_speed,RR,signal, high_speed,stop,yield1) == stop:
		label = "Stop"
		proba = np.exp(stop) / soft_sum
		label = "{}: {:.2f}%".format(label, proba * 100)
		print(label)
		print ("Stop")

	elif max (low_speed,RR,signal, high_speed,stop,yield1) == yield1:
		label = "Yield"
		proba = np.exp(yield1) / soft_sum
		label = "{}: {:.2f}%".format(label, proba * 100)
		print(label)
		print ("Yield")
	else:
		print("Nothing detected")
		
	frame = cv2.putText(frame, label, (10, 25),  
	cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)

	# show the output image
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break		

"""
# build the label
label = "Stop" if stop > notStop else "Not Stop"
proba = stop if stop > notStop else notStop
label = "{}: {:.2f}%".format(label, proba * 100)
"""
	# draw the label on the image
	#output = imutils.resize(orig, width=400)
	
print("[INFO] cleaning up...")
cv2.destroyAllWindows()
vs.stop()
