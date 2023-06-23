


"""# 2 IMAGE PREPROCESSING

2.1 Importing required libraries
"""

import cv2
import numpy as np
import os
from numpy import min , max
from google.colab.patches import cv2_imshow
from sklearn.preprocessing import minmax_scale
from pywt import swtn;             
from  glob  import glob;                   
from numpy import  dstack
from matplotlib.image import imread,imsave

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
import pandas as pd
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive')

"""2.2 NORMALISATION OF IMAGES"""

def normalization(image):
     Imax = max(image)
     Imin = min(image)
     return (image - Imin) / (Imax - Imin) ;

def startApp():
    SavePath = "/content/drive/MyDrive/covid-19/normalised normal" 
    ImagePath = "/content/drive/MyDrive/covid-19/normal/*.jpg"
    SWT_Level = 1
    AllImages = glob(ImagePath)
    print(AllImages)

    imagecount=0
    for imgName in AllImages:
        print(imgName)
        # Load the image
        image = cv2.imread(imgName);

        #Image Normalization
        image = normalization(image)
        print(np.array(image))
        print("imagecount")
        print(imagecount)
        imagecount=imagecount+1
        
        
        imsave(f'{SavePath}/{os.path.basename(imgName).replace("jpg","png")}',image)
if __name__ == "__main__":
    startApp()

    # [level4,extra]= swtn(image[:,:,3], wavelet="haar", level=SWT_Level, start_level=0, axes=None, trim_approx=True, norm=True)
    #rgb_image = dstack((level1,level2,level3,level4));

"""2.3 AUGMENTATION OF IMAGES"""

from numpy import fliplr , flipud , arange , asarray
from imutils import rotate_bound
from cv2 import imread , imwrite
from glob import glob
from scipy import ndimage

def image_affine_transform (image):
    height ,width , colors = image.shape
    transform = [[1,0,0],[0.5,1,0],[0,0,1]]
    return ndimage.affine_transform(image , transform ,offset = (0,-height//2 ,0),output_shape=(height , width+height //2 , colors))

def image_rotation(image,angle):
    rotated = rotate_bound(image, angle)
    return rotated

def h_flip(image):
    return fliplr(image)

def v_flip(image):
    return flipud(image)
   
images_path = '/content/drive/MyDrive/covid-19/abnormal/*.png'
augmented_path ='/content/drive/MyDrive/covid-19/augmentation' 

images=glob(images_path);
         
for image in images:
        image_number = 0 
        img=imread(image)
        for angle in arange(15,20,5):
            image_name = image.replace(".png","")
            img_rotate = image_rotation(img,angle)
            image_number += 1
            imsave(f'{augmented_path}/{os.path.basename(image_name)}aug{image_number}.png',img_rotate)
            img_hflip = h_flip(img)
            image_number += 1
            imsave(f'{augmented_path}/{os.path.basename(image_name)}aug{image_number}.png',img_hflip)
            img_aff_trans =image_affine_transform(img)
            image_number += 1
            imsave(f'{augmented_path}/{os.path.basename(image_name)}aug{image_number}.png',img_aff_trans)

"""2.3 IMAGE RESIZE TO 224*224"""

train_path= '/content/drive/MyDrive/FacialEmotionDataset/images/train'
val_path='/content/drive/MyDrive/FacialEmotionDataset/images/validation'

x_train=[]
for folder in os.listdir(train_path):

    sub_path=train_path+"/"+folder

    for img in os.listdir(sub_path):

        image_path=sub_path+"/"+img

        img_arr=cv2.imread(image_path)

        img_arr=cv2.resize(img_arr,(224,224))

        x_train.append(img_arr)


x_val=[]
for folder in os.listdir(val_path):

    sub_path=val_path+"/"+folder

    for img in os.listdir(sub_path):

        image_path=sub_path+"/"+img

        img_arr=cv2.imread(image_path)

        img_arr=cv2.resize(img_arr,(224,224))

        x_val.append(img_arr)

from google.colab import drive
drive.mount('/content/drive')

"""2.4 LABELING CLASSES AND SPLITTING DATA FOR TRAINING AND TESTING FOR VGG16 MODEL"""

train_x=np.array(x_train)
val_x=np.array(x_val)

np.save('/content/sample_data/images/train_x', train_x)

np.save('/content/sample_data/images/val_x', val_x)

train_x=np.load('/content/sample_data/images/train_x.npy')
val_x=np.load('/content/sample_data/images/val_x.npy')

train_path= '/content/drive/MyDrive/FacialEmotionDataset/images_augmented/train'
val_path='/content/drive/MyDrive/FacialEmotionDataset/images_augmented/validation'


from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255)
val_datagen = ImageDataGenerator(rescale = 1./255)


training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'sparse')


val_set = val_datagen.flow_from_directory(val_path,
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'sparse')

training_set.class_indices

train_y=training_set.classes
val_y=val_set.classes
train_y.shape,val_y.shape

"""# 3 Training and Testing Preprocessed FER Dataset using VGG16 CNN Model"""

# Input Image Size
IMAGE_SIZE = [224, 224]

vgg = VGG19(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

#do not train the pre-trained layers of VGG-19
for layer in vgg.layers:
    layer.trainable = False

x = Flatten()(vgg.output)

#adding output layer.Softmax classifier is used as it is multi-class classification
prediction = Dense(7, activation='softmax')(x)

model = Model(inputs=vgg.input, outputs=prediction)

# view the structure of the model
model.summary()

model.compile(
  loss='sparse_categorical_crossentropy',
  optimizer="adam",
  metrics=['accuracy']
)

from tensorflow.keras.callbacks import EarlyStopping
early_stop=EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=5)
#Early stopping to avoid overfitting of model

# fit the model
history = model.fit(
  train_x,
  train_y,
  validation_data=(val_x,val_y),
  epochs=10,
  callbacks=[early_stop],
  batch_size=32,shuffle=True)

# loss
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()

plt.savefig('vgg-loss-rps-1.png')
plt.show()

# accuracies
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()

plt.savefig('vgg-acc-rps-1.png')
plt.show()

model.evaluate(test_x,test_y,batch_size=32)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import numpy as np

y_pred=model.predict(val_x)
y_pred=np.argmax(y_pred,axis=1)

accuracy_score(y_pred,test_y)

print(classification_report(y_pred,test_y))

confusion_matrix(y_pred,test_y)

model.save("/content/drive/MyDrive/FacialEmotionDataset/vgg-rps-final.h5")

from keras.models import load_model
model = load_model('/content/drive/MyDrive/FacialEmotionDataset/vgg-rps-final.h5')

path="/content/sample_data/images"
for img in os.listdir(path):
    img=image.load_img(path+"/"+img,target_size=(224,224))
    plt.imshow(img)
    plt.show()
    x=image.img_to_array(img)
    x=np.expand_dims(x,axis=0)
    images=np.vstack([x])
    pred=model.predict(images,batch_size=1) 
    if pred[0][0]>0.5:
        print("angry")
        print(pred);
    elif pred[0][1]>0.5:
        print("disgust")
        print(pred);
    elif pred[0][2]>0.5:
        print("fear")
        print(pred);
    elif pred[0][3]>0.5:
        print("happy")
        print(pred);
    elif pred[0][4]>0.5:
        print("neutral")
        print(pred);
    elif pred[0][5]>0.5:
        print("sad")
        print(pred);
    elif pred[0][6]>0.5:
        print("surprise")  
        print(pred);
    else:
        print("Unknown")