
import numpy as np
from skimage import io, color, exposure, transform

import os
import glob
import h5py

from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization

from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras import backend as K
K.set_image_data_format('channels_first')

from matplotlib import pyplot as plt




NUM_CLASSES = 43
IMG_SIZE = 112

import pandas as pd
test = pd.read_csv('GT-final_test.csv', sep=';')

def preprocess_img(img):
    hsv = color.rgb2hsv(img)
    hsv[:,:,2] = exposure.equalize_hist(hsv[:,:,2])
    img = color.hsv2rgb(hsv)

    min_side = min(img.shape[:-1])
    centre = img.shape[0]//2, img.shape[1]//2
    img = img[centre[0]-min_side//2:centre[0]+min_side//2,
              centre[1]-min_side//2:centre[1]+min_side//2,
              :]

    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))

    img = np.rollaxis(img,-1)

    return img



def get_class(img_path):
    return int(img_path.split('/')[-2])


try:
    with  h5py.File('X.h5') as hf: 
        X, Y = hf['imgs'][:], hf['labels'][:]
    print("Loaded images from X.h5")
    
except (IOError,OSError, KeyError):  
    print("Error in reading X.h5. Processing all images...")
    root_dir = 'GTSRB/Final_Training/Images/'
    imgs = []
    labels = []

    all_img_paths = glob.glob(os.path.join(root_dir, '*/*.ppm'))
    np.random.shuffle(all_img_paths)
    print(all_img_paths)
    
    for img_path in all_img_paths:
        try:
    
            img = preprocess_img(io.imread(img_path))
            label = get_class(img_path)
            imgs.append(img)
            labels.append(label)

            if len(imgs)%1000 == 0: print("Processed {}/{}".format(len(imgs), len(all_img_paths)))
        except (IOError, OSError):
            print('missed', img_path)
            pass
    
    X = np.array(imgs, dtype='float32')
    Y = np.eye(NUM_CLASSES, dtype='uint8')[labels]

    with h5py.File('X.h5','w') as hf:
        hf.create_dataset('imgs', data=X)
        hf.create_dataset('labels', data=Y)


def cnn_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3),input_shape = (3,IMG_SIZE, IMG_SIZE), padding='same',
                      data_format='channels_first',
                     activation='relu'))
    model.add(Conv2D(32, (3, 3), padding='same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(4, 4),strides = 4))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), padding='same',
                     activation='relu'))
    
    model.add(Conv2D(128, (3, 3), padding='same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(4, 4),strides = 2))
    model.add(Dropout(0.2))

    model.add(Conv2D(512, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(512, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(512, (3, 3), padding='same',
                     activation='relu'))
    
    model.add(MaxPooling2D(pool_size=(7, 7),strides = 2))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(43, activation="softmax"))
    
   
    return model
    
lr = 0.01
sgd = SGD(lr=lr,decay = 1e-6,momentum = 0.9, nesterov = True)
def lr_schedule(epoch):
    return lr * (0.1 ** int(epoch / 10))

batch_size = 32
epochs = 30
model.compile(loss = "categorical_crossentropy", optimizer = sgd, metrics=["accuracy"])
history = model.fit(X, Y,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2,
          callbacks=[LearningRateScheduler(lr_schedule),
                     ModelCheckpoint('model_sgd_checkpoint.h5', save_best_only=True)]
          )
history.save('mon_sgd_final.h5')


X_test = []
y_test = []
i = 0
for file_name, class_id in zip(list(test['Filename']), list(test['ClassId'])):
    img_path = os.path.join('GTSRB/Final_Test/Images/', file_name)
    X_test.append(preprocess_img(io.imread(img_path)))
    y_test.append(class_id)

X_test = np.array(X_test)
y_test = np.array(y_test)

y_pred = model.predict_classes(X_test)
acc = np.sum(y_pred == y_test) / np.size(y_pred)
print("Test accuracy = {}".format(acc))


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.xticks(np.arange(0, 35, 5))
plt.yticks(np.arange(0.0, 1.0, 0.010))
plt.ylim(0.80, 1.0)
fig = plt.gcf()
fig.set_size_inches(20, 12)
fig.savefig('base_adam_callback_accuracy.png', dpi=100)

plt.show()



plt.plot(history.history['loss'])

plt.title('Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss'], loc='upper left')
plt.yticks(np.arange(0.0, 1.75, 0.10))
fig = plt.gcf()
fig.set_size_inches(20, 12)
fig.savefig('base_adam_callback_loss.png', dpi=100)
plt.show()

