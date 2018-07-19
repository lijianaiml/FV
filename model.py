from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.layers.core import Lambda
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model

# create the base pre-trained model
def finetune_model(encoding):
    base_model = InceptionV3(weights='imagenet', include_top=False,input_shape=(229, 229, 3))

    # add a global spatial average pooling layer
    x = base_model.output
    x = Flatten()(x)
    Lambda(lambda x: x -encoding)
    #x-=encoding
    # x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer

    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 1 classes
    predictions = Dense(1, activation='sigmoid')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='Adam', loss='binary_crossentropy')

    return model

# create the encoding pre-trained model
def base_encoding(image):
    model= InceptionV3(weights='imagenet', include_top=False,input_shape=(229, 229, 3))
    x=model.output
    x = Flatten()(x)
    base_model=Model(inputs=model.input,outputs=x)
    encoding = base_model.predict(image)
    return encoding

