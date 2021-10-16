
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import os
import model
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
        
val_datagen = ImageDataGenerator(rescale=1./255)

train_image_generator = train_datagen.flow_from_directory(
r'C:\Users\win10\task_2\data\train_images',
batch_size = 4)

train_mask_generator = train_datagen.flow_from_directory(
r'C:\Users\win10\task_2\data\train_masks',
batch_size = 4)

val_image_generator = val_datagen.flow_from_directory(
r'C:\Users\win10\task_2\data\val_images',
batch_size = 4)

val_mask_generator = val_datagen.flow_from_directory(
r'C:\Users\win10\task_2\data\val_images',
batch_size = 4)

train_generator = zip(train_image_generator, train_mask_generator)
val_generator = zip(val_image_generator, val_mask_generator)

NO_OF_TRAINING_IMAGES = len(os.listdir(r'C:\Users\win10\task_2\data\train_images'))
NO_OF_VAL_IMAGES = len(os.listdir(r'C:\Users\win10\task_2\data\val_images'))

NO_OF_EPOCHS = 30

BATCH_SIZE = 4

weights_path = r'C:\Users\win10\task_2'

m = model.FCN_Vgg16_32s()
opt = Adam(lr=1E-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

m.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics='accuracy')

checkpoint = ModelCheckpoint(weights_path, monitor='accuracy', 
                             verbose=1, save_best_only=True, mode='max')

csv_logger = CSVLogger('./log.out', append=True, separator=';')

earlystopping = EarlyStopping(monitor = 'accuracy', verbose = 1,
                              min_delta = 0.01, patience = 3, mode = 'max')

callbacks_list = [checkpoint, csv_logger, earlystopping]

results = m.fit_generator(train_datagen, epochs=NO_OF_EPOCHS, 
                          steps_per_epoch = (NO_OF_TRAINING_IMAGES//BATCH_SIZE),
                          validation_data=val_datagen, 
                          validation_steps=(NO_OF_VAL_IMAGES//BATCH_SIZE), 
                          callbacks=callbacks_list)
m.save('Model.h5')