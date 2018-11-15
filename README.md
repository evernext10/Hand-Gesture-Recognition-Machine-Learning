<h1 align="center"><B>Automatic method for the recognition of hand gestures for the categorization of vowels and numbers in Colombian sign language</B></h1>
<p align="justify">

This experiment developed a system which is designed to improve or facilitate communication between deaf people disability. The experiment has machine learning techniques to perform the due process of recognition of hand gestures of the Colombian sign language, recognizing the numbers from 0 to 5 and the vowels. 
This experiment works through 4 stages: Photo taking, pre-processing of the photo, extraction of the characteristics of the photo and finally performs the classification process for the identification of the gesture being performed. 
The image is captured by any camera that poses a good quality shot. Then, move on to the next stage of pre-processing, where you will for cleaning techniques to remove the shadow, the background and leave the image clean to perform the process of segmentation where the process of eliminating the noises that this pose takes place. In the feature extraction stage, it extracts the characteristics of the image that give us the mathematical methods like: Hu-moments, Fourier ellipticals, Histogram of oriented gradients (HOG) and Geometric characteristics. Finally, by means of the classifier of Multilayer perceptron Neural Network, Support Vector Machine and K-Nearest Neighbor it is obtained which the value of the sign, if it is a number or a vowel.

<p align="center"><img src="https://lh3.googleusercontent.com/-buKNTXAgGmbXj0h5-Mv8FWnJ9CisSNnccXtPOc40vSIVGipQqGuhEwAiu52PD6rLvaRD1qobTjty5gz5xV4S-SoGQ-wcrRFFd2glAC33r4Ja0wzHshlYVNKOyhsZUTX0lNcBHs8_J9iIc_CvXyU9zBMgPmMymd1R0Ujyn56LloeEeBDX4fqBEcgWTrbFyUGFaoR1DNoDlBRP8aU1LWgS3OtVscOQGJMswe-PbTNYNtVCpXirvu-eS3MxpZzeCQcwwUGNoNT_c1_UxTzeOrQO9VhOrLDpuMCOXxeP1GxMe_4er4JY_jq3IGoAzmlB9y_AlOw2B98eItP_1tkHUPaG5KzMDb-Fv8MmTOV_Ya1HExr7-xSPWHryMhBttnh1ta-PoncCBUy4w4wuYuiRZm4cBssgJ-68LCfG4uR363Qt6yT3J77llGcNnMZlAWDxpYVPXqx60loGVwyOBAavv1BVdYDEklZM5DEpU3WpNDYgQs3OflC9T74gajo3WiY_CBb8ztrhoRzvy23AakfyHH2uxM8lvX5Z0hZJ1wkaheayISJ_kZm2Y9wPOolILChp-Gavx5B-hukWaKZHBoW2R4OaF4svMd2LOeGhaVbzsK4r1G_tkf144__4rULmu1CFrky2HUc-61jCVYQAJwCRecX6UY7=w1462-h709-no">
  https://photos.app.goo.gl/8tvdcJYJdRHxcdYZ9
</p>

<h1 align="left"><B> First step: Dataset</B></h1>
<p align="justify">

In this stage is done a field work, which consisted of taking Photograf ́ıas using a C ́amara Fujifilm Finepix S4800, taking into account that for each ̃na were taken 3 photos of 3 different perspectives to the hand gestures of the vowels S and the n ́umeros from 0 to 5 of the LSC. The photographs were taken to people with different sizes of ̃nos, skin color and ambient lighting. Most of the photos were taken using flash and a small ̃naparte without flash, obtaining as a result a total of 3324 photos with resolution of 4608 x 2592 pixels in format .JPG

<p align="center"><img src="https://lh3.googleusercontent.com/GRAnLBiqV93ASdlP3flLgw66D6iItG10cVNqDCNsj0KyuTxySVfEwhtdGMeMTHd-hnqbuZV-WeP3Lpj5bHdlakPVk3WzpzXzRqyajX_KteWkfNqRvJcSRHFvc7q5A9pGDwm0zYFKzAu-YLAzg8PyI4kB7d2Q9hOF7IVxTq-9EA1CJtt37oWQC9zuyrAVqfumX2RF2mOYw0cxtH-P7XEWCb6SDWQt3Tut8wMbmBAL5Uus6lVWXwJv1XZCnVk_eIILALGfp_6GdFmANmNb4X6DzYA1lnSYmCyx3gEApoGgyv83yvNgLn-8-15gcoQCXk7ZKG86cwZ2xq2sndOltV4SzvizyIU-Wh-kq_vguT64ndGTP1u_7zngXDO6LL5gaJMaB-Khoi5M31PWGpLNguzm8A3QdhAX24Y95x1Ih2D4_Q-G8vIOtiRSu6WFFR2f71mD0PBrxbQmMC9MoVflloUZp0J7ucjiv1RPAc4h4R3OcpI-BhSChdBIEjbX85dpo4CbBwNttvfOx6wSgp7rJT67jfkjE353YKAp36Ay4-eD9vupynU8jlTXs33cpWvSVXqNp0rTJlqnLqVXJZdyKbEqxtS0hUL2b1YVHqW0o67DW3Hf7utYyA0bIxBcJK98MlYtBqLpiz96rPqa-b2dilet4zfm=w863-h709-no">
  https://photos.app.goo.gl/EhYS2sWFX1tfArZx8
</p>

<h1 align="left"><B> Second step: Preprocessing</B></h1>
<p align="justify">
  
In order to clean the dataset, data preprocessing techniques are applied, such as: Image resizing, RGB conversion to YCBCR, binarisation, erosion and finally filling of gaps this process is done so that in the next stage the process of Data mining is clean.

<p align="center"><img src="https://lh3.googleusercontent.com/dVIMvpbrh2MXIbsNKRn0HGHp63PL2z1bnMqFkWta7Wi3i6IcLZ18wrEQclNA4MX4Bi0nXiInC9jKz7wR_zAzGsx0OwfkFKbr96Vsp0MBsTLHLsimHbt6iNz5MQqdJajSMnvwMkR314t-WOFV_xH07WPjuWov32egkK0EgWYLwt9mAeRyjPdPYRI9FsHPFZnHmN5tZE1wf_71Z6UNmQBWDLa_6MzAugR-nKIKtVmwGs0Xv_PJ6KuOzQ59Ggl0Uu_TW3v0sXtdHqnZnQPBxi7KV4SwjJknNDFxqURsS3tyuk8nuCfHsRaplUIk4xku8vkeMhNuK1rb76fkEZGHYnmoLFoiy349uATCEPSbc1jpd4tB9hbzdaGlHkbP99so5BkHl26mRDcJBSqLOGryySvWgBUjuvvV5G5Lp5_YYbO-u7tFe3Zb-Bg5gLLznIpwfO87HD_eEP7F-LQh3NzjOpYpVhO9CSxV4xM1gcqVh23rH_NAkvMGPwaRgZM-PeIRsbLFp-qPf22Pcnwx3tI5cFLptzNuubIZtHCfxMlSItHROxVmdj_Mui1BTHKJtNjeW0_CUJBVH05i6pHjbcbZk0gKvbcLue7xenqSIccklszUjMbafml2RgiMISYFHNwGbk_sEFU7NAI2I8wl82UTeYFS6ee2=w902-h709-no">
  https://photos.app.goo.gl/fGNiT1tBzD441XYW9
</p>

<h1 align="left"><B>Third step: Feature Extractions</B></h1>
<p align="justify">
  
At this stage what is done is to represent numerically the image using the 4 methods used to extract characteristics: moments of Hu, histograms oriented to gradients, geometric characteristics and Fourier elliptic, obtaining as Result the corresponding numeric value characteristics vectors for each image. For each method used, a. txt document is generated with the image name followed by the characteristics vector and finally the tag.


