
# coding: utf-8

# In[ ]:


from skimage import data_dir,io,transform,color
import numpy as np

def convert_gray_batch(f):
    rgb=io.imread(f)    #依次读取rgb图片
    gray =color.rgb2gray(rgb)   #将rgb图片转换成灰度图
    dst=transform.resize(gray,(229,229))  #将灰度图片大小转换为229*229
    return dst
    
str='C:\\Users\\Administrator\\Desktop\\original\\n'+'\\*.pgm'
coll = io.ImageCollection(str,load_func=convert_gray_batch)
for i in range(len(coll)):
    io.imsave('C:\\Users\\Administrator\\Desktop\\Data\\Train\\Image\\2'+np.str(i)+'.jpg',coll[i])  #循环保存图片


# In[ ]:




