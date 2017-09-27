from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Activation, Dropout, Convolution2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential

batch_size = 32
nb_epoch = 10
nb_classes = 3
nb_train_samples = 7000
nb_validation_samples = 3000
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
model.add(Dense(nb_classes, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

train_datagen = ImageDataGenerator(rotation_range=40, width_shift_range=.2, height_shift_range=.2, shear_range=0.05, zoom_range=.1,horizontal_flip=True,fill_mode='nearest', rescale=1. / 255)

train_generator = train_datagen.flow_from_directory('/home/student/catkin_ws/src/CarND-Capstone/ros/src/tl_detector/light_classification/images/training', target_size=(64, 64), batch_size=batch_size, class_mode='categorical')

test_datagen = ImageDataGenerator(rotation_range=40, width_shift_range=.2, height_shift_range=.2, shear_range=0.05, zoom_range=.1,horizontal_flip=True,fill_mode='nearest', rescale=1. / 255)

test_generator = test_datagen.flow_from_directory('/home/student/catkin_ws/src/CarND-Capstone/ros/src/tl_detector/light_classification/images/validation', target_size=(64, 64), batch_size=batch_size, class_mode='categorical')

model.fit_generator(train_generator, epochs=nb_epoch, steps_per_epoch=nb_train_samples // batch_size , validation_steps=nb_validation_samples // batch_size, validation_data=test_generator  )

print(train_generator.class_indices)

model.save('traffic_light_classifier.h5')
