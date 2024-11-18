from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img
from PIL import Image
import numpy as np


image_path = 'test2.jpg'
model = load_model('Gender_and_Age.keras')
gender_dict = {0:'Male', 1:'Female'}


img = load_img(image_path, color_mode='grayscale')
img = img.resize((128, 128), Image.Resampling.LANCZOS)
img = np.array(img)
img = img.reshape(1, 128, 128, 1)
img = img/255.0
pred = model.predict(img)
pred_gender = gender_dict[round(pred[0][0][0])]
pred_age = round(pred[1][0][0])
print("Predicted Gender:", pred_gender, "Predicted Age:", pred_age)