from keras.layers import Conv2D, MaxPooling2D , LeakyReLU , BatchNormalization
from keras.layers import Dense, Dropout, Activation , Flatten
from keras.models import Sequential


def CNN_Model() :
    depth = 12    
    model = Sequential()
    model.add(BatchNormalization(input_shape=(48,48,1)))
    
    model.add(Conv2D(depth*1, (3, 3), activation = 'relu' , padding = 'same'))
    model.add(Conv2D(depth*1, (3, 3), activation = 'relu' , padding = 'same'))
    model.add(Conv2D(depth*2, (1, 1), activation = 'relu' , padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3) , strides = (2,2)))


    model.add(Conv2D(depth*2, (3, 3), activation = 'relu' , padding = 'same'))
    model.add(Conv2D(depth*2, (1, 1), activation = 'relu' , padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2) , strides = (2,2)))


    model.add(Conv2D(depth*3, (3, 3), activation = 'relu' , padding = 'same'))
    model.add(Conv2D(depth*3, (1, 1), activation = 'relu' , padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2) , strides = (2,2)))

    model.add(Conv2D(depth*2, (3, 3), activation = 'relu' , padding = 'same'))
    model.add(Conv2D(depth*2, (1, 1), activation = 'relu' , padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2) , strides = (2,2)))


#     model.add(Conv2D(depth*8, (3, 3), activation = 'relu' , padding = 'same'))
#     model.add(Conv2D(depth*16, (1, 1), activation = 'relu' , padding = 'same'))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D(pool_size=(2, 2) , strides = (2,2)))


    model.add(Flatten(name = 'Flatten'))
#     model.add(Dropout(0.2))
#     model.add(Dense(1024 , activation = 'relu'))
#     model.add(Dropout(0.2))
#     model.add(Dense(512 , activation = 'relu'))
#     model.add(Dropout(0.2))
#     model.add(Dense(256 , activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128 , activation = 'relu'))
    model.add(Dropout(0.2))
#     model.add(Dense(64 , activation = 'relu'))
#     model.add(Dropout(0.2))
    model.add(Dense(7, kernel_initializer='normal'))


    model.add(Activation('softmax', name='softmax1'))
    model.summary()
    return model


