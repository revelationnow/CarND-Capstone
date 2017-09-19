from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Activation, Dropout, Convolution2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential

batch_size = 32
nb_epoch = 10
nb_classes = 3
nb_steps = 204
image_shape = [64, 64, 3]

model = Sequential()

model.add(Convolution2D(32, (5, 5), padding="same", strides=(2, 2), activation="relu", input_shape=image_shape))
model.add(MaxPooling2D())
model.add(Dropout(rate=0.5, trainable=True, name="dropout_1"))

model.add(Convolution2D(64, (3, 3), padding="same", strides=(2, 2), activation="relu"))
model.add(MaxPooling2D())
model.add(Dropout(rate=0.5, trainable=True, name="dropout_2"))

model.add(Convolution2D(128, (2, 2), padding="same", strides=(2, 2), activation="relu"))
model.add(MaxPooling2D())
model.add(Dropout(rate=0.5, trainable=True, name="dropout_3"))

#model.add(Convolution2D(32, 2, 2,subsample =(1,1), border_mode="same", activation='relu'))
#model.add(MaxPooling2D())
#model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(1164))
model.add(Dropout(rate=0.5, trainable=True, name="dropout_4"))
model.add(Dense(512))
model.add(Dropout(rate=0.5, trainable=True, name="dropout_5"))
model.add(Dense(128))
model.add(Dropout(rate=0.5, trainable=True, name="dropout_6"))
model.add(Dense(nb_classes))
model.summary()

model.compile(loss='mse',optimizer='adam', metrics=['accuracy'])

datagen = ImageDataGenerator(width_shift_range=.2, height_shift_range=.2, shear_range=0.05, zoom_range=.1,fill_mode='nearest', rescale=1. / 255)

image_data_gen = datagen.flow_from_directory('/home/student/catkin_ws/src/CarND-Capstone/ros/src/tl_detector/light_classification/images', target_size=(64, 64), classes=['green', 'red', 'yellow'],batch_size=batch_size)

model.fit_generator(image_data_gen, epochs=nb_epoch, steps_per_epoch=nb_steps)

model.save('traffic_light_classifier.h5')
