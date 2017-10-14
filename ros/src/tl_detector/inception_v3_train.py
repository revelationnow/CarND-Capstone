import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
import pprint

start_model = InceptionV3(weights='imagenet', include_top=False,input_shape=(150,150,3), pooling='avg')

l2 = Dense(128, activation='relu', W_regularizer=l2(0.01)) (start_model.output)

# Logits
l3 = Dense(4,activation='softmax') (l2)

model = Model(input=start_model.input, output=l3)


#for layer in model.layers[:172]:
#    layer.trainable = False
#
#for layer in mode.layers[172:]:
#    layer.trainable = True

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


train_datagen = ImageDataGenerator(
        rotation_range=10,
        rescale=1./255,
        horizontal_flip=True,
        fill_mode='reflect'
        )

train_generator = train_datagen.flow_from_directory('data/train',
                                               target_size=(150,150),
                                               batch_size = 10,
                                               class_mode='categorical'
                                               )

test_datagen = ImageDataGenerator(
        rescale=1./255
        )

test_generator = test_datagen.flow_from_directory('data/test',
                                                  target_size=(150,150),
                                                  batch_size = 10,
                                                  class_mode='categorical'
                                                  )

pp = pprint.PrettyPrinter(indent=4)
print("Printing training classses : ")
pp.pprint(train_generator.classes)
pp.pprint(train_generator.class_indices)
print("Printing validation classses : ")
pp.pprint(test_generator.classes)
pp.pprint(test_generator.class_indices)


checkpoint = ModelCheckpoint('inception_v3_checkpoint.hdf5',
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True,
                             mode='max',
                             period=5)
model.fit_generator(train_generator,
                    steps_per_epoch=200,
                    epochs=25,
                    validation_data=test_generator,
                    validation_steps=40,
                    callbacks=[checkpoint]
                    )

#model_json = model.to_json()
#with open(model.json) as json_file:
#    json_file.write(model_json)
#
#model.save_weights('inception_site.h5')
model.save('inception_v3_site.h5',)
print("Saved model to file")




