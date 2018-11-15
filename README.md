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

<p align="center"><img src="https://lh3.googleusercontent.com/y-wOt4qpbct1mOz6pQ8tl932cYjMuvNyOhQzXNhIxjEHAwKGaAWnztnHKXXxGHtApFqXphtx9u2r9J08vtgIBFX-ikDQEFc_p7TQUk5YRGaZFbVDDvO1l0vsNduD4h3OOcnRhEfuW0Q5_WcVHtfVGWk7l0GxJ_xx7mxcNPK6B0oOfI8A6IBU-kKKdvPWztLcFcYfy0y8nc4WfbNxlOJDMMLRMhICGFkY3B1ojlqiW96eS0YmhWQshkHhpLmWPVg9PMY0IVj18xyv6oGKLpNQswE9dg7wXC8ZejRqq9AoWe1ICrf49SCMa-55NmYExdWH7qFSTEc0kFd3EoVGvUCbDiq9ge1VBtVlPeIkAzP4fgOQrajZ_dffN_dUUlkSK8yjte0cffQxsdJB9YEeKpXIaovO9-bgOvEWEI5LDC6zOHjrivAXdnzM5WVPPid-72XZ_MAUfpcAVIgOmCIKGPemuImqK6mZTIxfa9kUdKxe0yV8nJYfLXsI1IADalO9R4Lmco1DSKcCzFbncDqQfQUAiVHRYk30wsEr_q0sK-ORw532WjE61cWZ_nUOVeV-Y6fzYMzThHGgvJLQA3dtL9frVCmftv7CFGdAjlKsxg5Vlvj0qoh_nhzUWzKV1rSou2rxf46c-iYlw2TE2agDo5YBCYvl=w584-h577-no">
  https://photos.app.goo.gl/tJYim9w7Fymg31449
</p>

For the classification with Neural Network, the following hyper-parameters were used:


Parameter  | Second Header
------------- | -------------
ACTIVATION  | ['identity','logistic','tanh','relu']
SOLVER  | ['lbfgs']
LEARNING_RATE_INIT  | [0.0001]
HIDDEN_LAYER_SIZES  | [(100, 1), (100, 2), (100, 3)]

<p align="center"><img src="https://lh3.googleusercontent.com/KZaWOSFQx9GAcQM8f-Hc6iJjwOTBxyL-hfvsDFLZARuFzorUuoSll_onTJOLHpbEryHTPtdbe6-V8Py5urHi5bEN7Ux7BllcZItkot5wa9fo5f9keagqE1togpBxA-pg-fuYiqkY00-LoRmwC0oihUDJ1xX9pQAx-9_6k4A8u0-U794cekVYRS7EbgKbAFMUt_sP1ITye7HFdyj_Pqls6pP61T7WbUPsUk2HkETyN7ia5Sva84uzkTakrzzCcBCL0MTFuPMt6jEvFVRJ69G_vnXpEfPyNT4Dz4jg3dC7c512ByHnv7YgTb6PC-VIa1Bkvm6GaZVCDDcmt28eVTiQoW7GHYrNFbRUYyHLYnNlaD6YKcr7x1G-koJCR2O0NPNuVZ7SgSW8qyRsYK_mxIBCGqyu4UsFGCx-Xodb9ioHct4vrJRpOJDYo7Y7D_bh7qzcjodnDGVPleMQkKJgV0QLKKlJnpBeQgrSxX6VQ8v2SZmV1t_tav1JB21PrNQLrvawoVIJMFBVNtEeMN4imxQ95SS5spgdbckHi_uc5z-mm1MLq7Us6BAuSCC0Nj6xBmr5pY97iTpzpKaO8VpPy2lftipyjhdgYx2gLCNBCcQTy0RQZ7VE5i6U546CdotHGjmb3RAUkBRa2FMS-vZ9UmWNLph7=w969-h494-no">
  https://photos.app.goo.gl/CMpRy1aQexy1QCsq9
</p>

For the classification with K-Nearest Neighbors, the following hyper-parameters were used:


Parameter  | Second Header
------------- | -------------
N_NEIGHBORS  | [1,2,4,6,8,10]
ALGORITHM  | ['auto']
WEIGHTS  | ['uniform', 'distance']
N_JOBS  | [-1]

<p align="center"><img src="https://lh3.googleusercontent.com/qqgJWCLN-lH1gSGA2DOM1ERjPEDJIRGcTlKUYLo9sCoDNPu3ubOLqf4Mj-5O6ZEpB3bBbrCX3IdxMoj7u57_Po1wFMXDJynEjOMRHzY8dGyehmb71qJiKBwbN0Pg03fLlQbPT_USgV2svWdlpQ6vFDgYih7luYbwrFukDSZWZ441TsLYMxJU-v4OKpKnfSY3jDOKQ7F-v2MY2v77EuOZmfxpCmgnezb87_GPM3tmM57SoGHg0yowa_cqxKzu3l-QMDFOG4wBKp921suevgHjAR9TisfSOkcwFRjGwaHhl1SB43tm8ptImZmpqJcoTokRF-hNmCN-4QgiORFtLg_f9Ma6fMZa8vHx02IDKGv1IhEC5EwTjI3UqC21gJzLXZ-nW0A30nWKHyaJ07fg57Hqk25l9024nIrQYa1UXAHKLMa-Vz03B_8MuI35Y7nXbRT8X6AKPCAzsG7bbuR7X6ZTcjKvs7rsqZde4C83pH4sgrMcc4lFs9-OPN2kfdaZ8f7552bYqVSjsTnW1rKPSfiipLVHYevfalycjgzIxEaBatlnUOWdQ_62DiVIWuZP4rooBazyWZNLqICE8hWmmXtc2vMY2Ynk4B6NQYmoDxKEMPBQpTlijC_3lOPoJ74LZLz94KexpOq8dmDnunhByG_F3lTl=w747-h603-no">
  https://photos.app.goo.gl/iRLjPgWM78wWdyxe9
</p>

<h1 align="left"><B>Results</B></h1>
<p align="justify">
  
The results show where it is divided by training percentages and testing according to the characteristics compared to the performance results of precision, recall and F1-score using the vector support machines, also highlights the best result That was obtained using that classifier.  
According to table 2, it can be seen that the best method for vector support machines is the gradient-oriented histograms with the Fourier ellipticals with a percentage of 70 %, 69 %, 69 % accuracy, recall and F1-score respectively using 80 % of Training set and 20 % test set.

Being:  
P: Precision, R: Recall and F1: F1-score
EF: Elliptic Fourier
HOG: gradientes-oriented histograms
HOG-PCA: Gradient-oriented histograms with main componentes analysis
Hu: Hu Moments

<p align="center"><img src="https://lh3.googleusercontent.com/hRcPb5rEMKbzf3RsMjTvEzUrDEnWHymU7cxZv_MJnsx2iiyO4yfXq8MYF9MuT1zov2yVvkGyt8oOyFWNWLlxk6gW2fHbEB_LyCF7wG2iL6nIfRfSUVCeNioMnKtP0-_1XxEd1LBmQALRN3sVHkQXrTiJemoKZ2PrVxWtJcJgdQYv1vb0UfyMxRG1xrIQU5Ar39VapFaJEbxu51lEnkW12gO6DGpclZGf74PV9x0O82ruYbfsidpBlDEq32cOEthn91f-sAwRDOq__nbEwTXl7NwVK2CVaNl_iLzKv4VjRvl3pyKarnbiGNhqThhy0q1FWYu2MD1c9jLaup1BhnmWHoEay1hajYOmLsq7NRnHLjgo0gt-RfbFDKSuOTitfl4Bq29nmZMsnim4adr3vJRgYoJmkPBbhfOd4Tzto2_JLKlxUqGVncty3fKd7mDLUtyeHhzHkNJHT57-NT0IWMER9JDxluK0SvrrHa6W0UEozZ0bU4rA1Ya6U8CVlpMVWSnH3nB_vEAoPW3srW2DorByw1J7R59s4E4QHuHfJWbeZVjaL00Ct4ltoXerrfvFnZeVlaU9DGHqK720u2yD-0I0RT3uAjxZBfDZT3mKugttpqaijBSgKciwNOCVaHJ5n8N84AprCDVVE-ovV5DnvwTpMYNP=w576-h498-no">
  https://photos.app.goo.gl/gCQdjaAkZDm7xwTx9
</p>
