
# coding: utf-8

# In[ ]:


from PIL import Image
def convert_gray():

    I = Image.open('.\\data\\2.jpg')
    I.show()
    print(I.size)
    L = I.convert('L')
    L.show()
    L = L.resize((229,229),Image.ANTIALIAS)
    L.show()
    print(L.size)
    return L
convert_gray()

