# USAGE
# python stop_detector.py 

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

# define the paths to the Not Santa Keras deep learning model and
# audio file

MODEL_PATH = "C:/Users/ddabl/Documents/computer/DM/assignments/auton/saved_modelv1.model"
#MODEL_PATH = "stop_not_stop.model"
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

# load the model
print("[INFO] loading model...")
model = load_model(MODEL_PATH)

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

# loop over the frames from the video stream
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
    
    #initialize since our data is not binary
    #this will have to be updated once we move to multi class 
    notStop = 0

    # classify the input image and initialize the label and
    # probability of the prediction
    #(notStop, stop) = model.predict(image)[0]
    low_speed, RR, signal, high_speed, stop, yield1 = model.predict(image)[0]
	
    soft_sum = np.exp(low_speed) + np.exp(RR) + np.exp(signal) + np.exp(high_speed) + np.exp(stop) + np.exp(yield1) 

    if max (low_speed,RR,signal, high_speed,stop,yield1) == low_speed:
        label = "Low"
        notStop = 1
        #proba = np.exp(low_speed) / soft_sum
        #label = "{}: {:.2f}%".format(label, proba * 100)
        #print(label)
        print ("Low")

    elif max (low_speed,RR,signal, high_speed,stop,yield1) == RR:
        label = "RR"
        notStop = 1
        #proba = np.exp(RR) / soft_sum
        #label = "{}: {:.2f}%".format(label, proba * 100)
        #print(label)
        print ("RR")

    elif max (low_speed,RR,signal, high_speed,stop,yield1) == signal:
        label = "Signal"
        notStop = 1
        #proba = np.exp(signal) / soft_sum
        #label = "{}: {:.2f}%".format(label, proba * 100)
        #print(label)
        print ("Signal")

    elif max (low_speed,RR,signal, high_speed,stop,yield1) == high_speed:
        label = "High"
        notStop = 1
        #proba = np.exp(high_speed) / soft_sum
        #label = "{}: {:.2f}%".format(label, proba * 100)
        #print(label)
        print ("High")

    elif max (low_speed,RR,signal, high_speed,stop,yield1) == stop:
        label = "Stop"
        proba = np.exp(stop) / soft_sum
        label = "{}: {:.2f}%".format(label, proba * 100)
        print(label)
        print ("Stop")

    elif max (low_speed,RR,signal, high_speed,stop,yield1) == yield1:
        label = "Yield"
        notStop = 1
        #proba = np.exp(yield1) / soft_sum
        #label = "{}: {:.2f}%".format(label, proba * 100)
        print(label)
        print ("Yield")
    
    label = "Not Stop"
    proba = notStop

	# check to see if stop sign was detected using our convolutional
	# neural network
    if stop > notStop:
		# update the label and prediction probability
        label = "Stop"
        proba = stop

		# increment the total number of consecutive frames that
		# contain stop
        TOTAL_CONSEC += 1

		# check to see if we should raise the stop sign alarm
        if not STOP and TOTAL_CONSEC >= TOTAL_THRESH:
			# indicate that stop has been found
            STOP = True
            print("Stop Sign...")

	# otherwise, reset the total number of consecutive frames and the
	# stop sign alarm 
    else:
        TOTAL_CONSEC = 0
        STOP = False

	# build the label and draw it on the frame
    label = "{}: {:.2f}%".format(label, proba * 100)
    frame = cv2.putText(frame, label, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

	# show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
print("[INFO] cleaning up...")
cv2.destroyAllWindows()
vs.stop()

