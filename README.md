
Fraud Analytics : Real or Fake Image Detection 
Project Overview
Our project for the Fraud Analytics subject focused on developing a machine learning model using Python to classify images as real or fake. The primary objective was to leverage pre-trained models through transfer learning and implement them on image datasets consisting of real and tampered images. The final output was a deployable solution that provides a user-friendly interface for image classification, including additional metadata generation.

Dataset Link : https://www.kaggle.com/datasets/sophatvathana/casia-dataset/data

Methodology
Data Collection and Preprocessing
The image dataset used in the project contained two categories: real (authentic) and fake (tampered) images. The images were resized to match the input size of the pre-trained models.
We used standard image augmentation techniques to ensure the robustness of the model.


Model Selection and Transfer Learning
We experimented with two pre-trained image classification models: VGG16 and ResNet. VGG16 Model:
We applied transfer learning by freezing the initial layers of the VGG16 model and fine-tuning the fully connected layers to adapt to our dataset.
After training the model on our dataset, we achieved an accuracy of approximately 85%.
The model was saved in the .h5 format after the transfer-learning process for further deployment. ResNet Model:
We also experimented with the ResNet model, but the accuracy was significantly lower at 65%, which led us to finalize the VGG16 model for deployment.


Deployment on Streamlit
To make the model accessible and user-friendly, we deployed it on Streamlit, a web framework that allows interactive user interfaces.
App.py File:
We created an App.py file to handle the model's deployment and define necessary functions.
The app provides a simple UI where users can upload an image, and the model processes it to classify whether the image is real or fake.
Functionality:
A.	Image Upload: Users can upload an image through the UI.
B.	Classification: The image is processed by the VGG16 model to detect if it's real or fake.
C.	Metadata Generation: Along with the classification, the app generates metadata that includes:
1.	File size
2.	File path
3.	Edge Detection: The app also displays an edge-detected version of the uploaded image as part of the analysis.
 
Conclusion
Through this project, we successfully implemented a robust fraud detection system for image classification using transfer learning. The VGG16 model outperformed ResNet in terms of accuracy, making it our final choice. The model has been deployed through Streamlit, offering an intuitive and interactive user interface. The deployment not only classifies images but also generates additional metadata for further analysis.
This project demonstrates the efficacy of using pre-trained models for image classification and shows how AI-driven fraud detection can be practically implemented in real-world scenarios.
Deployment
The Link to Streamlet: imagedetector-xgbmsr6npzxwun249appure.streamlit.app
User Interface

![image](https://github.com/user-attachments/assets/9a724207-9ce3-4cf9-b717-405783f3f72d)


Uploaded an image to detect whether it’s fake or real.
Additionally, metadata for the image is generated along with an edge-detected version of the image.
 
 ![image](https://github.com/user-attachments/assets/613869e1-102f-4904-8c50-418bcce6ca5e)
 ![image](https://github.com/user-attachments/assets/a5317f22-4425-4cff-a0b5-43ff1e71acd0)
![image](https://github.com/user-attachments/assets/955863ea-cac1-4244-93b4-a00d443f4501)
![image](https://github.com/user-attachments/assets/5b0367fe-60d8-45b4-a92c-d6f8d166a935)
![image](https://github.com/user-attachments/assets/4e8f914d-5f08-4216-8ecf-6268e2d9a06a)
![image](https://github.com/user-attachments/assets/13bf9baa-da67-467e-93ca-4c2d8f6affdf)


 
 Streamlit Link: https://imagedetector-xgbmsr6npzxwun249appure.streamlit.app/ 
