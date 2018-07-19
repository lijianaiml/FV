
# coding: utf-8

# In[ ]:


from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
import numpy as np
from keras.layers.core import Lambda
import pandas as pd

from keras.applications.inception_v3  import preprocess_input
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential



from sklearn.metrics import log_loss
from skimage import data_dir,io,transform,color
from model import base_encoding,finetune_model


#load path
train=pd.read_csv('C:\\Users\\Administrator\\Desktop\\Data\\Train\\labelc.csv',lineterminator='\n',header=0)
train_path="C:\\Users\\Administrator\\Desktop\\Data\\Train\\Image\\"
me_path="C:\\Users\\Administrator\\Desktop\\Data\\Train\\me\\me.jpg"


image_me=image.load_img(me_path,target_size=(229,229))
image_me=image.img_to_array(image_me)
image_me=preprocess_input(image_me)

train_img=[]
#batch deal train_img start
def dealimage(f):
    temp_img=image.load_img(f,target_size=(229,229))
    temp_img=image.img_to_array(temp_img)

    return temp_img
    
str='C:\\Users\\Administrator\\Desktop\\Data\\Train\\Image\\'+'\\*.jpg'
coll = io.ImageCollection(str,load_func=dealimage)
for i in range(len(coll)):
    train_img.append(coll[i])
train_img=np.array(train_img) 
train_img=preprocess_input(train_img)
#batch end

train_y=np.asarray(train['label\r'])


#切分训练集验证集
from sklearn.model_selection import train_test_split
X_train, X_valid, Y_train, Y_valid=train_test_split(train_img,train_y,test_size=0.1, random_state=42)

# Load our model
image_me=image_me.reshape(1,229,229,3)
encoding = base_encoding(image_me)
model = finetune_model(encoding)
#model.summary()



batch_size = 1 
nb_epoch = 20
# Start Fine-tuning
model.fit(X_train, Y_train,batch_size=batch_size,epochs=nb_epoch,shuffle=True,verbose=1,validation_data=(X_valid, Y_valid))

# Make predictions
predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)

# Cross-entropy loss score
score = log_loss(Y_valid, predictions_valid)
print(score)

