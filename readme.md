## PiCar Project

This is a model for a small raspberry pi autonmous car.
It recognises several traffic signs.

### Phase 1,2 and 3

The building of the car:

Images (to be added):

*  [Image 1](https://drive.google.com/open?id=1qRFprUwxraU2BONuyHZT7aT9OYGDUnbf)
*  [Image 2](https://drive.google.com/open?id=1oT72TggzrbR4Aw_tmEFDYx2deOoa65sN)
*  [Image 3](https://drive.google.com/open?id=12dgRKMX1zmIlBoy3wx8eTngdm-s84P2h)
*  [Image 4](https://drive.google.com/open?id=1hFmEudmus4cbYBgzzCiL2EyNFc8mRSym)
*  [Image 5](https://drive.google.com/open?id=1dDChY1-8zsRxvuJOfN65APJ-vnWSsEwt)
*  [Image 6](https://drive.google.com/open?id=1PvUKLVOz_7GUTqXRzz_edYRtFrNch2wd)
*  [Image 7](https://drive.google.com/open?id=1rNLfBQPUlSaoCCnhoaWHh7g_praXsLuH)
*  [Image 8](https://drive.google.com/open?id=1ZCq8LtoERXHhM-eFJSi7FaxTiDZ70iNc)
*  [Image 9](https://drive.google.com/open?id=11HHeoyQrSXIRbSPK4a78_NjNQN1QT0RM)


## Dataset

* Image sie: 28x28
* Number of Stop_Sign images:
* Number of RailRoad images:
* Number of Yield_Sign images: 
* Number of High_Speed images
* Number of Low_Speed images
* Number of Signal images: 

Setting up the OS and configuring it to be used.

## Phase 4

### CNN Architecture:

![CNN Architecture] (signs_model.png)
Creating the Neural Network model to recognise stop sign initially.

However, the model for this group recognises 6 signs.

1. Stop
2. Yield
3. Low speed
4. High speed
5. RailRoad
6. Signal (traffic light)

Links (to be added):
1. [Recognise Traffic Signs](http://www.youtube.com/watch?v=6qRq6aZwnzw "Sign Recognition")
2. [Lane Detection with PiCar](http://www.youtube.com/watch?v=vaN8VT8Z0qA "Lane Detection")
3. [Moving Car and Stop Sign](http://www.youtube.com/watch?v=IbNAn3VLDZg "Moving Car and Stop Sign")

## Suggestions on limitations

* HDMI port should not be stuck behind servo
* Protective shell for exposed circuits
* Points for improvement on lane detection:

* Curved line detection would be beneficial, as I think the code only handles straight lines.
* I had thought the use of going into the HSV color space and boosting a particular color hue to enhance the line detection.
* The camera should be made use of to do scan if it is not finding 2 lanes.
* I found that optimizing the OpenCV library itself can lead to a significant improvement. See this link:
* https://www.theimpossiblecode.com/blog/build-faster-opencv-raspberry-pi3/

### Points for improvement on controlling wheels:

* I think the movement of the car should be left to while loops to ensure smooth movement. E.g. whilst lanes are detected and car is aligned properly then keep moving forward.
* The car should also stop if no lanes are detected.
* Speed should also be dynamically adjusted. 

