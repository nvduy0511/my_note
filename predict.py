from tensorflow.keras.models import load_model
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import os

#labels = ['black', 'blue', 'brown', 'green', 'orange', 'pink', 'red', 'silver', 'white', 'yellow']
labels = ["beige","black","blue","brown","gold","green","grey","orange","pink","purple","red","silver","tan","white","yellow"]
exp_th = 2

model = load_model('models/Xception-car color-87.98.h5')

path_img_test = 'D:/PROJECTS/AI/Color/Data_vehicle/DATA/NguyenTrai-NguyenQuyen-P2'
images_test = os.listdir(path_img_test)
path_save = 'D:/PROJECTS/AI/Color/Data_vehicle/images_test/runs'
path_save += f'/exp_Avinh{exp_th}'
enable = True
last_file = '17_58_40_614__25987__1301521__98C23928_midtruck___0.8263263__origin_obj.jpeg'
for image in images_test:
    if image == last_file:
        print("OKE")
        enable = True
    if enable:
        img_ori = Image.open(os.path.join(path_img_test, image))
        img = img_ori.crop(( 10,  10, img_ori.width -  10, img_ori.height -  10))
        img = img.resize((299, 299)) # Resize the image to match your model's input size
        img_array = np.array(img)
        #img_array = img_array.astype('float32') / 255.0 # Normalize pixel values to [0, 1]
        img_array = np.expand_dims(img_array, axis=0) 
        predictions = model.predict(img_array)
        # Get the predicted class index and class name
        predicted_class_index = np.argmax(predictions[0])

        folder_save = os.path.join(path_save, labels[predicted_class_index])
        if not os.path.exists(folder_save):
            os.makedirs(folder_save)
        img_ori.save(os.path.join(folder_save,image))
        print(image)