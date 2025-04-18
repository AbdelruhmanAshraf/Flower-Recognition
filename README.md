# Flower-Recognition

---

# Flower Species Classification with CNN and Flask Web App

This project is aimed at classifying flower species from images using a Convolutional Neural Network (CNN) model built with TensorFlow and Keras. The model is trained using the EfficientNetB3 architecture and can classify flower images into five categories: daisies, dandelions, roses, sunflowers, and tulips.

## Project Structure

- **Jupyter Notebook**: Contains the code for data preprocessing, model training, and evaluation.
- **app.py**: A Flask web application that allows users to upload flower images and get predictions from the trained model.
- **model.h5**: The saved trained model that can be loaded and used for predictions in the Flask app.

## Requirements

Make sure you have the following dependencies installed:

- Python 3.x  
- TensorFlow 2.x  
- Keras  
- OpenCV  
- Flask  
- Matplotlib  
- Seaborn  
- scikit-learn  
- PIL (Pillow)

## Dataset

The dataset used in this project is the **Flowers Dataset** from Kaggle. It contains images of flowers in the following categories:  
- Daisy  
- Dandelion  
- Rose  
- Sunflower  
- Tulip  

You can download the dataset from this link:  
**[Dataset Link](https://www.kaggle.com/datasets/rahmasleam/flowers-dataset)**

## Data Preprocessing

1. The images are resized to 224x224 pixels to fit the EfficientNetB3 model's input requirements.  
2. Data augmentation techniques (rotation, zoom, horizontal flip) are applied to increase the diversity of the training set.  
3. The dataset is split into training, validation, and testing sets using an 80/20 split for training and testing.

## Model Architecture

The model uses the **EfficientNetB3** architecture with the following layers:  
- **Base model**: EfficientNetB3 (pre-trained on ImageNet)  
- **BatchNormalization** layer  
- **Dense** layer with 256 units and ReLU activation  
- **Dropout** layer with a rate of 45% to avoid overfitting  
- **Final Dense** layer with 5 units (for 5 flower classes) and a softmax activation function

## Web App (Flask)

The web application allows users to upload images and get predictions from the trained model.

### How to Run the Flask App

1. Make sure to place the `model.h5` file in the same directory as `app.py`  
2. Run the Flask app using:  
   `python app.py`  
3. Open the app in your browser at: `http://127.0.0.1:5000/`  
4. Upload flower images via the interface to get predictions.

### API Endpoints

- **POST /**: Upload an image and receive the prediction as a JSON response.


## Model Evaluation

After training the model, its performance is evaluated on the validation and test sets using:  
- Accuracy  
- Precision  
- Recall  
- AUC (Area Under the Curve)

### Example Output

For a sample image of a flower, the model might return:

```json
{
  "prediction": "sunflower",
  "confidence": 0.92
}
```

## Logs

All app activities (uploads, predictions, errors) are saved in a file named `app.log` to track performance and bugs.

## Contact

For any questions, suggestions, or collaborations:

**Name**: Abdelrahman Elfekky  
**Email**: abdelruhamanelfekky@gmail.com
**GitHub**: https://github.com/AbdelruhmanAshraf 

---
