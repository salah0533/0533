class _3D_UNet_2dCONV_tensorflow():

  def __init__(self,inputs,ker_init,dropout):
    self.inputs = inputs
    self.ker_init = ker_init
    self.dropout = dropout

  def build_unet(self):

    #Encoder
    #input  150x150x200x4
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = self.ker_init)(self.inputs) #output 150x150x200x32
    conv1 = Conv2D(32,3, activation = 'relu', padding = 'same', kernel_initializer = self.ker_init)(conv1) #output 150x150x200x32

    #input  150x150x200x32
    pool2 = MaxPooling2D(pool_size=(3, 2))(conv1) #output 150x50x100x32
    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = self.ker_init)(pool2) #output 150x50x100x64
    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = self.ker_init)(conv2) #output 150x50x100x64
    
    #input  150x50x100x64
    pool3 = MaxPooling2D(pool_size=(1, 2))(conv2) #output 150x50x50x64
    conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = self.ker_init)(pool3) #output 150x50x50x128
    conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = self.ker_init)(conv3) #output 150x50x50x128
    
    #input  150x50x50x128
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv3) #output 150x25x25x128
    conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = self.ker_init)(pool4) #output 150x25x25x256
    conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = self.ker_init)(conv4) #output 150x25x25x256
    
    #latent space
    #input  150x25x25x256 ---
    pool = MaxPooling2D(pool_size=(5, 5))(conv4) #output 150x5x5x256
    conv = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = self.ker_init)(pool) #output 150x5x5x512
    conv = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = self.ker_init)(conv) #output 150x5x5x512
    drop = Dropout(self.dropout)(conv) #output 150x5x5x512
    #------------------------
  
    #Decoder
    #input  150x5x5x512
    up7 = UpSampling2D(size = (5,5))(drop) #output 150x25x25x512
    conv7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = self.ker_init)(up7) #output 150x25x25x256
    merge7 = concatenate([conv4,conv7], axis = 3) #output 150x25x25x512
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = self.ker_init)(merge7) #output 150x25x25x256
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = self.ker_init)(conv7) #output 150x25x25x256

    #input  150x25x25x256
    up8 = UpSampling2D(size = (2,2))(conv7) #output 150x50x50x256
    conv8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = self.ker_init)(up8) #output 150x50x50x128
    merge8 = concatenate([conv3,conv8], axis = 3) #output 150x50x50x256
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = self.ker_init)(merge8) #output 150x50x50x128
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = self.ker_init)(conv8)  #output 150x50x50x128

    #input  150x50x50x128
    up9 = UpSampling2D(size = (1,2))(conv8) #output 150x50x100x128
    conv9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = self.ker_init)(up9) #output 150x50x100x64
    merge9 = concatenate([conv2,conv9], axis = 3) #output 150x50x100x128
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = self.ker_init)(merge9) #output 150x50x100x64
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = self.ker_init)(conv9)  #output 150x50x100x64
    
    #input  150x50x100x64
    up10 = UpSampling2D(size = (3,2))(conv9) #output 150x150x200x64
    conv10 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = self.ker_init)(up10) #output 150x150x200x32
    merge = concatenate([conv1,conv10], axis = 3) #output 150x150x200x64
    conv = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = self.ker_init)(merge) #output 150x150x200x32
    conv = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = self.ker_init)(conv)  #output 150x150x200x32
    
    #input  150x150x200x32
    conv10 = Conv2D(4, (1,1), activation = 'softmax')(conv) #output 150x150x200x4
    
    return Model(inputs = self.inputs, outputs = conv10)