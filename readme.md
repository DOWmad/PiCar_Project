## PiCar Project

This project has the following requirements:
* Lane detection
* Train a Neural Model to recognise traffic signs (up to 6)
* Drive down a track and correct itself if it deviates.

### Phases 1 -  3

The building of the car:

Images:

Car Build:

*  [Image 1](https://drive.google.com/open?id=1qRFprUwxraU2BONuyHZT7aT9OYGDUnbf)
*  [Image 2](https://drive.google.com/open?id=1oT72TggzrbR4Aw_tmEFDYx2deOoa65sN)
*  [Image 3](https://drive.google.com/open?id=12dgRKMX1zmIlBoy3wx8eTngdm-s84P2h)
*  [Image 4](https://drive.google.com/open?id=1hFmEudmus4cbYBgzzCiL2EyNFc8mRSym)
*  [Image 5](https://drive.google.com/open?id=1dDChY1-8zsRxvuJOfN65APJ-vnWSsEwt)
*  [Image 6](https://drive.google.com/open?id=1PvUKLVOz_7GUTqXRzz_edYRtFrNch2wd)
*  [Image 7](https://drive.google.com/open?id=1rNLfBQPUlSaoCCnhoaWHh7g_praXsLuH)
*  [Image 8](https://drive.google.com/open?id=1ZCq8LtoERXHhM-eFJSi7FaxTiDZ70iNc)
*  [Image 9](https://drive.google.com/open?id=11HHeoyQrSXIRbSPK4a78_NjNQN1QT0RM)
*  [Image 10](https://drive.google.com/open?id=1mZ2oYhBbblA1ZFDlyoyjrI_eOFc1driL)

Software config:

*  [Image 1](https://drive.google.com/open?id=11-FAVVyjcwZH6j3xSx2mT9xkuXvrgT4X)
*  [Image 2](https://drive.google.com/open?id=1GrOu_MUWyIbRqUHSRJmp20WTU0aTA5tE)
*  [Image 3](https://drive.google.com/open?id=1q59am1R3MTlamauAMxLex52hXFZpv7NG)
*  [Image 4](https://drive.google.com/open?id=1zpfwEh3MtUunQ5er0yifwfffbrEfK6QZ)
*  [Image 5](https://drive.google.com/open?id=1Cqjww_A_s0t5WupSc1Eeri0NRBArim1m)
*  [Image 6](https://drive.google.com/open?id=1Y--OcvCqFFAqWhDnqQ_4zV9wQOA77662)
*  [Image 7](https://drive.google.com/open?id=1JWNsaig1Heg_kB34I9PlYaocBIfuxb6)
*  [Image 8](https://drive.google.com/open?id=1ZuromBNZU6EixK3SA39m17BIJlCE16Gu)
*  [Image 9](https://drive.google.com/open?id=1rqPYRZnEoBUHbzVvpTqL3dIlZ0xD2vlp)
*  [Image 10](https://drive.google.com/open?id=1MM3JT71xHO-q5cV_ld4imqXzcrvzG8qM)

## Dataset

* Image size: 28x28
* Total number of images: 2550


## Phase 4

### CNN Architecture:

![CNN Architecture](https://github.com/DOWmad/PiCar_Project/blob/master/signs_model.png)
![Short CNN Model](https://github.com/DOWmad/PiCar_Project/blob/master/CNN_model_short.png)

### Training

![Training Results](https://github.com/DOWmad/PiCar_Project/blob/master/Training_Results.png)

Creating the Neural Network model to recognise stop sign initially.

However, the model for this group recognises 6 signs.

1. Stop
2. Yield
3. Low speed
4. High speed
5. RailRoad
6. Signal (traffic light)

Video Clip Links:
1. [Recognise Traffic Signs](http://www.youtube.com/watch?v=6qRq6aZwnzw "Sign Recognition")
2. [Lane Detection with PiCar](http://www.youtube.com/watch?v=vaN8VT8Z0qA "Lane Detection")
3. [Moving Car and Stop Sign](http://www.youtube.com/watch?v=IbNAn3VLDZg "Moving Car and Stop Sign")
4. [Testing car drive with signs](https://youtu.be/fZW2c-99Lec "Testing car drive with signs")

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


### Questions

####1. Image Size

The size of the image can improve the training of the Neural Network. 
Simply by cropping the image to the region of interest means there is a smaller size and less time is spent to distinguish unrelated objects in the image. Therefore, when considering hundreds of images for testing and training this can make it a big difference quicker both in terms of accuracy and speed.
This should then also help the CNN kernels extract features.

#### 2. How to design CNN architecture including how many layers, what kind of layers, and so on
Building a CNN is about being efficient and effective. It is also a challenge to speculate on how many layers and what kind of CNN is best, because of variations based on the intended purpose. In general, there are no strict rules to be adhered to, or a true default to follow. In many cases, trial and error is required before the correct CNN design is built. Common factors involved in a CNN are: an input layer, at least two CNN layers, a max pooling layer, loss calculation then output. There is also the filter which is part of the convolution, dropout layers and rectification layers. A good approach is to review other, well established CNN such as DenseNet: the premise of a dense CNN is that it would reference feature maps from previous layers of the network, which increases variation in the input of subsequent layers [1]. 

#### 3. How to optimize the model including parameter values, drop out, back propagation, learning rate, # of epoch and so on

A model trained for an image-recognition CNN would benefit from the improvement of the training data: As the network is fed training data, it improves its parameters by using both stochastic gradient descent (including optimizers such as Adam) and back propagation with each epoch. Optimization of a CNN would benefit from recognizing the present limitations. Since training a CNN on a common laptop or desktop computer is largely out of the question due to lack of CPU capability to handle the datasets, an idea to optimize would be to create several train/validate sets from multiple, similar datasets and then run more epochs on smaller sets. Back propagation can come into play to do the “learning” of the network by taking the loss and calculating the implication of each layer with respect to the loss to then adjust the weights to improve the loss.
Choosing the optimal learning rate is crucial since if it is too low then the learning will take far longer and it is too high then it may fail to converge.

#### 4. Evaluations
To evaluate the performance of the CNN, parameters for accuracy need to be set. The development of a train/validation pair of data is generally what is used. 
The use of logarithmic loss during training is advised, and then running the trained CNN through a Confusion Matrix will help evaluate the true performance of the model [2].

#### 5. How to overcome the limitations in your DM-Car implementation

After building the car and achieving connection to it, past software errors the biggest limitation in our DM- Car implementation is lack of power. 
To train our CNN for a desired high accuracy rate, we would need to train it using a data set of thousands—hundreds of thousands of images perhaps, and we do not have sufficient processing power to handle the task. Further, the RaspberryPi itself is a low-powered device with limited capabilities. 
These limitations might become significant drawbacks with logarithmic loss and a lower efficiency of back propagation during testing.
By being careful and optimizing as much as possible and generating a trained model with high accuracy we can help ensure the RaspberryPi is utilized effectively.

