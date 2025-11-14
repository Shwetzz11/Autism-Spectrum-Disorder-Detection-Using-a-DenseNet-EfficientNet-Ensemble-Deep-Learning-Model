Project title: 
Autism Spectrum Disorder Detection Using a DenseNet–EfficientNet Ensemble Deep Learning Model

Abstract -autism spectrum disorder is a neurodevelopmental disorder characterized by a deficit in social communication, behavior, and emotion recognition. The earlier the diagnosis of ASD is accomplished and the more accurately it is done, the more effective the treatment will be, but the traditional diagnostic techniques are somewhat subjective, time-consuming, and usually need expert opinion. This work proposes an automatic classification technique for ASD using facial images through the deep learning-based ensemble of the networks DenseNet121 and EfficientNetB0. Our model integrates the features from both networks with a feature-fusion mechanism that leverages their unique strengths complementarily to improve learning. We further propose a custom preprocessing pipeline involving Gaussian smoothing followed by face alignment, equalization of brightness and contrast, and real-time data augmentation. The model was trained and tested on the public dataset available for Autism Image Data, which gave a test accuracy of 78.67%, ROC-AUC of 0.8726, and an F1-score of 0.80, thus showing consistency in performance to identify ASD-related facial cues. Experimental results demonstrate that our ensemble captures ASD-related subtle facial cues really well and forms a key step toward non-invasive computer-assisted early screening systems.

Team Members:
Shweta V 
Samyuktha S 
Dhanya S 

Base paper reference: 
T. Farhat, S. Akram, M. Rashid, A. Jaffar, S. M. Bhatti, and M. A. Iqbal, “A deep learning-based ensemble for autism spectrum disorder diagnosis using facial images,” PLoS ONE, vol. 20, no. 9, pp. 1–15, 2025.

Tools and libraries used: 
All experiments were conducted in Google Colab using a T4 GPU runtime environment that provided efficient deep learning computation. The environment was set up with TensorFlow 2.x and other essential Python libraries required for images pre-processing, model training, and performance evaluation. The hardware comprises an NVIDIA Tesla T4 GPU with 16 GB VRAM and Google Drive storage that allowed for persisting the models and datasets. The software environment is Ubuntu 20.04, running Python 3.10 on TensorFlow-Keras 2.x, supported by libraries including NumPy, OpenCV, SciPy, scikit-image, Matplotlib, Seaborn, scikit-learn, and Dlib.

Steps to execute the code:
•	Open the Notebook
Access the project notebook directly via Google Colab using the link below:
•	Configure the Runtime Environment
Navigate to Runtime → Change runtime type → Hardware accelerator → GPU, select T4 GPU, and click Save to enable hardware acceleration.
•	Mount Google Drive
When prompted, authorize access to Google Drive so that the notebook can load datasets. 
•	Install Dependencies
All required packages are listed in the requirements.txt file.
The notebook automatically installs them using the following command block:
!pip install -r requirements.txt
•	Execute the Pipeline
Run all cells sequentially (Runtime → Run all) to perform dataset preprocessing, model training (EfficientNetB0 + DenseNet121 ensemble), validation, and visualization.
•	Review the Outputs
Training/validation accuracy and loss graphs are displayed inline. Confusion matrix, performance metrics, and final classification results are generated automatically. Trained model weights and plots are saved to the designated /models and /results directories. 

 Description of dataset: The data utilized in the study was acquired from the freely accessible Kaggle Autism Image Data Repository: https://www.kaggle.com/datasets/cihan063/autism-image-data
 It contains facial images of autistic and non-autistic individuals in the age range 2–14 years. Before preprocessing, this dataset was divided in the following way AUTISTIC with 1240 images and NON-AUTISTIC with 1240 images. Total 2480 images.

Output screenshots or result summary: 
The proposed dual-input ensemble model, for which EfficientNetB0 and DenseNet121 served as the base, proved to be very consistent in its performance and reliable for classifying face images of both ASD and non-ASD subjects. During training over 45 epochs, it reached a maximum training accuracy of 82.50%, while the maximum validation accuracy attained was 89.58% at epoch 19, indicating good generalization of the model with controlled overfitting. Confirmation of stability and robustness in the learning process is further provided by higher validation accuracy compared to training accuracy. Later, the top three models were ensembled to generate a final consensus prediction, keeping in view the complementarities in feature representations extracted from both networks. This translates, after postprocessing through ensembling, into an overall classification accuracy of 78.67%, ROC-AUC of 0.8726, PR-AUC of 0.8853, and F1-score of 0.80 on the held-out test set. This constitutes 128 true positives and 108 true negatives, representing a sensitivity of 85.33% and a specificity of 72.00%. These confirm that the model has good reliability in pinpointing ASD-related facial features with adequate false-positive control.
 
YouTube demo link: https://youtu.be/LBAH5UMAvn0

