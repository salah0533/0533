import tensorflow
from keras.layers import Conv2D,MaxPooling2D,Dropout,concatenate,Input,UpSampling2D
from keras import Model

# make sure your in the same directory of the file
from _3D_UNet_2dCONV_tensorflow import _3D_UNet_2dCONV_tensorflow


X = tensorflow.ones((150,150,200,4))
model = _3D_UNet_2dCONV_tensorflow(input_layer, 'he_normal', 0.2,Model, Conv2D,MaxPooling2D,Dropout,concatenate,UpSampling2D)

model = model.build_unet()
print(model.summary())

p = model.predict(X)
print(f'output shape: {p.shape}')