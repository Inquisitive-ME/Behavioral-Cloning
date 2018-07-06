# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./WriteupResourses/InceptionLayer.png "Inception Layer Visualization"

[FinalDriving]: ./WriteupResourses/AutonomousProvidedData.gif "Autonomous Driving - Final Result Trained from provided data"
[ReluAllData]: ./WriteupResourses/AutonomousReluAllProvidedData.gif "Autonomous Driving - Relu Activations Trained from provided data"
[ReluSelectedData]: ./WriteupResourses/AutonomousReluSelectedProvidedData.gif "Autonomous Driving - Relu Activations Trained from selection of provided data"
[1LapBackwards]: ./WriteupResourses/Autonomous1Backwards.gif "Autonomous Driving - Trained from 1 recording of backwards Lap"
[1LapForwards]: ./WriteupResourses/Autonomous1Forwards.gif "Autonomous Driving - Trained from 1 recording of forwards lap"
[BothLaps]: ./WriteupResourses/Autonomous1LapEach.gif "Autonomous Driving - Trained from combining forwards and backwards laps"

[Track2]: ./WriteupResourses/AutonomousTrack2.gif "Autonomous Driving on Track 2"

[SideImage]: ./WriteupResourses/SideImage.png "Sample Side Image"
[CroppedSideImage]: ./WriteupResourses/SideImageCrop.png "Cropped Sample Side Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* AutonomousDriving.mp4 video of the car driving with collected data

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model was based on the architecutre defined in the paper [Going deeper with convolutions](https://arxiv.org/pdf/1409.4842v1.pdf) with a single inception layer. It consisted of a 7x7 convolution followed by a max pooling layer, followed by a 3x3 convoluiton and another max pooling layer. This then fed into an inception layer as shown in the picture below. The inception layer went through 1 fully connected layer before going to a single output (model.py lines 106-140) 

![alt text][image1]

I used Linear activation functions instead of non-linear relu activation functions mainly because it created a smoother steering output. I will discuss this in greater detail later. The data is normalized in the model using a Keras lambda layer (model.py line 109). 

Below is a table describing the layers of the architecture

| Layer Name 						| Input			| Output Size	 	| Parameters 	|
| --- 								| ---				| ---				| ---			|
| InputLayer						| Image			| (160, 320, 3)		| 0			|
| PreProcessing - Normalize Image	| InputLayer		| (160, 320, 3)		| 0			|
| Cropping (Cropping2D) 			| PreProcessing 	| (85, 320, 3)		| 0			|
| conv2d_1 - 7x7 strides = 2			| Cropping		| (40, 157, 64)		| 9472		|
| max_pooling2d_1 - 3x3 strides = 2 	| conv2d_1		| (19, 78, 64)		| 0 			|
| conv2d_2 - 3x3 linear activation		| max_pooling2d_1 | (17, 76, 192) 	| 110784		|
| max_pooling2d_2 - 3x3 strides = 2	| conv2d_2		| (8, 37, 192)		| 0 			|
| conv2d_3 - 1x1 linear activation		| max_pooling2d_2 | (8, 37, 64)		| 12352		|
| conv2d_4 - 3x3 linear acticvation	| max_pooling2d_2 | (8, 37, 128)		| 221312		|
| conv2d_5 - 5x5 linear activation		| max_pooling2d_2 | (8, 37, 32)		| 153632		|
| max_pooling2d_3 - 3x3 strides = 1	| max_pooling2d_2 | (8, 37, 192)		| 0			|
| concatenate_1 (Concatenate)		| conv2d_3 <br> conv2d_4<br> conv2d_5<br> max_pooling2d_3 |  (8, 37, 416)		| 0			|
| dropout_1 40%					| concatenate_1	| (8, 37, 416)		| 0			|
| activation_1 linear				| dropout_1		| (8, 37, 416)		| 0			|
| flatten_1 (Flatten) 				| activation_1 		| (123136)		| 0			|
| dense_1 (Dense)	elu activation		| flatten_1		| (10)			| 1231370	|
| dense_2 (Dense) 					| dense_1		| (1)				| 11			|
 
============================================================================
Total params: 1,738,933
Trainable params: 1,738,933
Non-trainable params: 0
__________________________________________________________________________________________________

#### 2. Attempts to reduce overfitting in the model

I don't feel that I had a huge issue with overfitting. When I was using relu activation layers I did have some overfitting issues.The main issue was that the model would train well and have a low mean squared error, but then would completely miss some turns. I fixed this by excluding some of the data. I also had better results using linear activation functions. With linear activations my main issue was that I would find local minima where the output was just a constant steering value. Excluding data also helped with this issue. This didn't solve the problem, and when I ran into this issue I would retrain the model until it found a better minima or use a stochastic gradient decent optimizer where I could define the learning rate to try to step over the local minima.

The model contains a dropout layer in order to reduce overfitting (model.py lines 131). 

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

My main tuning parameters were involving what training data to include. I have the following parameters (model.py lines 30-39):
 
 * pUseImage - chance of including the center image
 * pIncludeSide -chance of including the side images
 * pFlipSide - chance of including flipped side images 
 
If the center image is selected I always included the flipped image. I also found that I could get good results without having to flip the side images.
I set these probabilites seperate for samples where the steering angle is 0 and samples where the steering angle is non-zero. Most of the data had a steering angle of zero so I found that it was helplful to exclude some of these data points to allow the data where the car is steering to have more of an impact. 

I also had a steering_correction factor which I did not alter much from a value of 0.25. This was the value that was added to the steering angle for the left images and subracted from the steering angle for the right images

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 143). I did occassionally use a stochastic gradient decent optimizer, where I adjusted the learning rate to get better results.

As mentioned before, I also started with relu activations. I ran into some overfitting issues, but my main probablem was the steering command was very jittery and occassionaly on a model that had low mean squared error the car would still miss turns. I was able to get a much smotther command using linear activations, and did not have issues with turns. But this came at the cost of having issues of the model training to a constant steering angle. My final decision was to use linear activation functions and an elu activation function on the last layer.

#### 4. Appropriate training data

Initially I trained my model with the provided data, which I found harder to get good steering behavior with then when I created my own data by driving around the track once in each direction.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My overall strategy was to start with an already defined architecture and train it for the driving data. I really liked the inception idea of GoogLeNet so I decided to start with the architecutre defined in the paper [Going deeper with convolutions](https://arxiv.org/pdf/1409.4842v1.pdf) with a single inception layer.

You can see the architecture of the model used in the above section "An appropriate model architecture has been employed"

I started with just the front facing camera from the provided data. I found that I was able to get a low mean squared error of about 0.09, but when I deployed the model to drive in the simulator the car would end up off the road and not able to recover. To fix this I tried two tactics.

1. I tried manipulating the data so that more images of the car steering were included and less images of the car drivig straight were included.

2. I included the side camera images in the data

I had some success including less data where the car was driving straight, but ultimately I had to include the side camera images to get the car to maintain the center of the road.

I also decided to to flip the center camera images along with the steering command and include this as additional data. I felt this would help the system be more robust. I did not flip the side camera images as this did not seem to have a huge benifit and increased training time.

With all the center images included and flipped, and also including the side images with a steering offset I had issues with the car missing some turns and my steering was very jittery. Below is a GIF of the car driving with the same architecture but with relu instead of linear activations, trained on the data as described above.

![alt text][ReluAllData]


I figured out that by excluding some of the images through tuning my parameters below, I was able to get smoother steering performance.

```python
    if(steering_center == 0):
        #no steering
        pUseImage = 0.75
        pIncludeSide = 0.85
        pFlipSide = 0.0
    else:
        #steering
        pUseImage = 1.0
        pIncludeSide = 0.65
        pFlipSide = 0.0
```
 
Below is a GIF of the car driving with a model trained with less images of when the steering angle was 0.

![alt text][ReluSelectedData]

I decided that I still wanted to get smooter performance. I found that from changing my activation functions to linear, besides the final activation being elu, along with the tuning of what images to include in training I was able to get much smoother performance. Below is the final results from using the provided data.

![alt text][FinalDriving]

Using the linear activations I did run into some issues where the model would train to a constant steering command. Reducing the images used for training helped reduce this, but ultimatley I was able to tell that if the training results did not get below a 0.02 mean squared error I needed to retrain the model until it did.


#### 2. Creation of the Training Set

Next I decided to see how well my model would perform on data I collected. I found this task to be much less challenging. So to become a little more challenging I decided to try to train on just data captured from driving 1 lap in the reverse direction. Below is a GIF of the results. As you can see the car does make it around the track but has some trouble spots. In order to get this to work I had to adjust the data that was used to train the model using my tuning parameters described above. Below are the settings

```python
    if(steering_center == 0):
        #no steering
        pUseImage = 0.75
        pIncludeSide = 0.85
        pFlipSide = 0.0
    else:
        #steering
        pUseImage = 1.0
        pIncludeSide = 1.0
        pFlipSide = 1.0
```

![alt text][1LapBackwards]

I then trained the model on just one lap driving in the same direction. This performed better as can be seen from the GIF below.

![alt text][1LapForwards]

Finally I used both of these datasets together to train the model. So below is a GIF of the model trained on 1 lap going the correct direction and 1 lap going the reverse direction.

![alt text][BothLaps]

As you can see I did not have too many issues with just using this data and did not need to include training data of the car recovering from the side of the road.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 3. PreProcessing of Image

I used images from both the front and side cameras. The first step I did was normalize the image by dividing the data by 255 and then subtracting 0.5

Then I cropped the image by removing 50 pixels from the top and 20 from the bottom. This was to exclude the scenary and the car to prevent the model from adjusting to them in the training.

Below is a picture of the original image, followed by a picture of a cropped image.

![alt text][SideImage]

![alt text][CroppedSideImage]

I also augmented the data by flipping some of the images as well as the angles thinking this would help to make the model more robust and also provide more training data.

I finally randomly shuffled the data set and put 20% of the data into a validation set to test for overfitting.

### Additional Comments
Ultimatley I felt like tweaking the provided data with linear activations to get a smooth steering performance, got me the best results. However as I switched to the second track I found that the linear activations were even more challenging to get a good result, and found that the relu activations handled the additional data and track situations better.

* I did try using a generator, but it was much slower than just loading all the data into memory. This was mainly because I have a GTX 1080 graphics card that I am using for the training. This was extremely benificial to allow me to quickly train and test the model.

* I also attempted track 2, I was not able to get fully around the track, but was suprised at how well the model was able to do with just 2 laps of training data. I will have to put this on my todo list of projects to work on if I get bored

![alt text][Track2]
