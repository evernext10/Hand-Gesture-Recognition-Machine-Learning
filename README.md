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

<p align="center"><img src="https://lh3.googleusercontent.com/GoesxLpl1Ehc-QOfwr9su07T8cB7nRE6t37RAprHrntP2QVy5CtmVxYzXwTX1JFjy1wKl2onpk9Qc0q97BfCK4ZvuSSH3qcrskVcCMlt56nps2JrFQDf-2yPQInKYNw9AcqMdPUxWKazDAB3hZV0Pxew6FskGgRL_TS9fFbCBN6TTWgXuyO71FX2Abapgaqc9hFTPmsZ29OakY9mEX0MmRqCCmRHX81h5Ld3TFtg_ibW8PApih3ljjYK8556bnjuny16S_N_Dd42cjwsq6xzMCMmyV6fy6cnFvF_FaOBCo3SjAmPc7AM4hZVZ8dr53Sc0AG_nhqkYYj7lOatTZSxFXx7o4ukGRTSxEmySh2ccm0A9HeiwiYqPhu4oXaciula4Mn4FqZXmpnuSD28m03YocLvYdXD-FrlL4f4mnb0S9OoeXhNWQeFEO8JHOmlgJY4Mld1s9CtI7JTYfJLimdSSTz2UkqH7N3hwxlVfL4kH_-X9KzubOTsoBdcHI0l10XjOZYPkUk9rVmOmYxr8pLvPfJgODkI4ozJFESkHKN1q6LgAm9-D7UEOFW8JE76ozCOLZwkCi32iqtTmPGOHdQHf9XiDzY1aJGIYhgWyXQwxnkTtfUqcVvd-_rHaqq6RNOiTEsFN2FmQuCw1E3k0k3lEFsT=w991-h707-no">
  https://photos.app.goo.gl/3SUFPWf6E4vxiKeQ9
</p>

After getting the features we store them in. txt files:

<p align="center"><img src="https://lh3.googleusercontent.com/LriTsTtLuMcjfzzCsQkqEWzHoNVTEWSvkSzopXvd4fjd_CELKKXNJFA5AmFW93TWQd5hPL1N5YbVg01bIHM055QyHrHegGWXjHG8ELP5CShPqdRTdjoKl9LMcnWe_H6OQ6KNEuKMyevL4pp10S07fj6yFdGqKXt4RXHhNVcHngxiW0vk7Q715P87P2oAMRiCHITISaxm6QYhh5I7fcIg-yo4qjnoTXi3QoSBDXqnDbTqAC0Z-bdFSG4XkAH9M0W0daZy54TsaKSA7g0iwxA9C3PTF-Lko9HYjqsUb6ul-i2CLgxbm--wgo6YHLcBiNCILgdpMI03NXrW6u-Q_tGC-UKJ2CpsiAtORVH8DjWwKO7PIdeDPgH48e-Jj_nCMmdX3wqHsiJT6CEmuC7V3a_or73auNuS5rN76YTlqLWCrNxMKboxBSrO-QFvLDlcJYO17f0uRxe1ZLueiiDcJEJ2-IiITAbtqBm7mLBxTm-DvgkvHINOq0TlqSbCkUnioyrvozNu1uacNwuOA9cG6HqR2hCHNiPsH2q310X0AsGPx3aPrvTt-TtH_zk8sYgmpRe67_QwsF0LIgRFXhf0fyW-4oyokaH7_NMTSKfXfU4sm1D9ijiwiP2ErB8gW6kxQ57uwHPR97pXXzPUq4xkDbA8YGYM=w786-h133-no">
  https://photos.app.goo.gl/pQD4LcochbsHBvxx6
</p>

<h1 align="left"><B>Fourth step: Sampling</B></h1>
<p align="justify">
 
At this stage sampling is carried out using cross-validation of K-folds using 5 pages, the data is divided with percentages of 70 %, 75 % and 80 % to train the algorithm and percentages of 30 %, 25 %, 20 % for the test set, respectively, of each folio is Gets a validation score and finally calculates the average of the scores.

<p align="center"><img src="https://lh3.googleusercontent.com/ojHNqTy0mLJ0Ah1dMstdm0OCnrjXLCFIdQgJGgv1V29BLAACcobZ3KMUYhPwaEaQRkD1Te_9qfgFt0vY3laK0y_95xYBljG5n7NYLqMwjNRlUdAW4Bbjdk3FP4GAwa9et4cRmo55oOioB5B4LP0LQZ4KleRBuQ-gh_71BwHHTqtm4QjbwdZGUSAk8v4NISz0FPVdawwKNOrUdo8vTnfRCYGJs90zoimCql8Il4nYIua4iAm9qzJoLF3-n3q61kJAJR7-pXP4Qe28C9x1TtPPaj1Rm8Qa6oza3antDn1X44zb1vVq6mTThto9RbzpH2lDKr1pd0_n3lXPWb6TcpAB2c2QTijBkuMKeoe53egVvOwY7dVJiLYkrE9I9E0K2CQ1NQwwIE-lzjxhFzvrfFlDUQnOap80MwOyYWNH75yUwLppwpcuek4aGmvnirzvlAsHWm32Vv6mzHyys1gjNK6EZrN0NguxKmCTdGQF18NSy6pu7Sn_zKGoX14Q4_NGW7EUWxmh1gP06fWAcKVDGTrQBZTqqO7VJT4-vN3djN3hIF0WCgEEU2ciUw5aC26DPd6mE0UJnxj9OEpy6zTCYAgo2VxLuaMwqCa9hZezigXcIgM3393cxrNp2e6rptCfcHyaaMzFXxWsFwTJ0_rW3T6-gd7z=w1221-h656-no">
  https://photos.app.goo.gl/rf9rrNQW6Fm8iR7h8
</p>
  
<h1 align="left"><B>Fifth step: Classification</B></h1>
<p align="justify">
  
 The grading stage consists of two key units, the feature extraction unit and the Pattern rating unit. 
First, characteristics such as Hu moments, gradient-oriented histograms, Fourier ellipticals and geometric characteristics are extracted. Applying PCA (principal component analysis) to HOG to take only the most relevant or important features.
Following the pattern classification unit is implemented using the vector support machine method.
Finally the patterns are recognized and they are classified in their different classes.

For the classification with Support Vector Machines, the following hyper-parameters were used:


Parameter  | Second Header
------------- | -------------
KERNEL  | rbf
GAMMA  | 0.0001, 0.001, 0.01, 0.1, 0.2, 0.5
C  | 0.01, 0.1, 1, 10, 100, 1000

For the classification with Neural Network, the following hyper-parameters were used:


Parameter  | Second Header
------------- | -------------
ACTIVATION  | ['identity','logistic','tanh','relu']
SOLVER  | ['lbfgs']
LEARNING_RATE_INIT  | [0.0001]
HIDDEN_LAYER_SIZES  | [(100, 1), (100, 2), (100, 3)]

For the classification with K-Nearest Neighbors, the following hyper-parameters were used:


Parameter  | Second Header
------------- | -------------
N_NEIGHBORS  | [1,2,4,6,8,10]
ALGORITHM  | ['auto']
WEIGHTS  | ['uniform', 'distance']
N_JOBS  | [-1]
