import os


import pandas as pd
import xml.etree.ElementTree as ET 

import time 
import math 
import cv2 
import numpy as np 
from PIL import Image
from imgaug import augmenters as iaa 
import matplotlib.pyplot as plt
import matplotlib.image as image 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,f1_score, confusion_matrix, accuracy_score
import tensorflow as tf 
from keras.utils import np_utils
from keras.utils import Sequence    
from keras import layers 
from keras.preprocessing.image import ImageDataGenerator 
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# MODEL ARCHITECTURE SPECIFIC LIBRARIES
from keras.applications.vgg16 import VGG16 , decode_predictions, preprocess_input
from keras.applications.resnet_v2 import ResNet152V2 , decode_predictions, preprocess_input
from keras.applications.resnet_v2 import ResNet50V2 
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense,BatchNormalization,LayerNormalization
from keras.models import Sequential
from keras.utils.vis_utils import plot_model
from keras import Model
from openpyxl import Workbook
from convert_report2excel import convert_report2excel

np.random.seed(42)
tf.random.set_seed(42)

image_path = 'DataSetStore/Images'
image_size = 224
batch_size = 32

breed_list = sorted(os.listdir(image_path)) # sorted list of all files and directories in path (in this case 120 directories, each containing images of a specific dog breed)
num_classes = len(breed_list) # number of dog breeds
print("{} breeds".format(num_classes))

# define time counter function to test algorithm performance

_start_time = time.time()

def process_time_starts():
    global _start_time 
    _start_time = time.time()

def time_elapsed():
    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec,60)
    (t_hour,t_min) = divmod(t_min,60) 
    print('The process took: {}hour:{}min:{}sec'.format(t_hour,t_min,t_sec))


label_maps = {}
label_maps_rev = {}
for i, v in enumerate(breed_list):
    label_maps.update({v: i})
    label_maps_rev.update({i : v})

def paths_and_labels():
    paths = list()
    labels = list()
    targets = list()
    for breed in breed_list:
        base_name = "./data/{}/".format(breed)
        for img_name in os.listdir(base_name):
            paths.append(base_name + img_name)
            labels.append(breed)
            targets.append(label_maps[breed])
    return paths, labels, targets

paths, labels, targets = paths_and_labels()

assert len(paths) == len(labels)
assert len(paths) == len(targets)

targets = np_utils.to_categorical(targets, num_classes=num_classes)

class ImageGenerator(Sequence):
    
    def __init__(self, paths, targets, batch_size, shape, augment=False):
        self.paths = paths
        self.targets = targets
        self.batch_size = batch_size
        self.shape = shape
        self.augment = augment
        
    def __len__(self):
        return int(np.ceil(len(self.paths) / float(self.batch_size)))
    
    def __getitem__(self, idx):
        batch_paths = self.paths[idx * self.batch_size : (idx + 1) * self.batch_size]
        x = np.zeros((len(batch_paths), self.shape[0], self.shape[1], self.shape[2]), dtype=np.float32)
        y = np.zeros((self.batch_size, num_classes, 1))
        for i, path in enumerate(batch_paths):
            x[i] = self.__load_image(path)
        y = self.targets[idx * self.batch_size : (idx + 1) * self.batch_size]
        return x, y
    
    def __iter__(self):
        for item in (self[i] for i in range(len(self))):
            yield item
            
    def __load_image(self, path):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        image = cv2.resize(image, (224, 224))  # Resize image
        
        if self.augment:
            # Define augmentation sequence
            seq = iaa.Sequential([
                iaa.OneOf([
                    iaa.Fliplr(0.5),
                    iaa.Flipud(0.5),
                    iaa.MultiplyBrightness((0.2,1.2)),
                    iaa.Sometimes(0.5,

                    ),
                    iaa.Affine(
                        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                        rotate=(-40, 40),
                        shear=(-8, 8)
                    )
                ])
            ], random_order=True)
            
            # Apply augmentation
            image = seq.augment_image(image)
        
        # Normalize pixel values
        image = image.astype(np.float32) / 255.0
        
        return image


x_train, x_test, y_train, y_test = train_test_split(paths, targets, test_size=0.1, random_state=42) #10% test
x_train, val_x, y_train, val_y = train_test_split(x_train, y_train, test_size=0.1, random_state=42) #10% validate

train_ds = ImageGenerator(x_train, y_train, batch_size=32, shape=(image_size, image_size,3), augment=True)
val_ds = ImageGenerator(val_x, val_y, batch_size=32, shape=(image_size, image_size,3), augment=False)
test_ds = ImageGenerator(x_test, y_test, batch_size=32, shape=(image_size, image_size,3), augment=False)

base_model = ResNet152V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers[:-10]:
    layer.trainable = False
# base_model.summary()

def feature_extractor(x):
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    return x


def classifier(x):
    x = Dense(2048, activation='relu')(x)
    x = Dense(2048, activation='relu')(x)
    x = LayerNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(512, activation='relu')(x)
    return x


def prediction_layer(x):
    return tf.keras.layers.Dense(num_classes, activation='softmax')(x)

x = base_model.input
x = feature_extractor(x)
x = classifier(x)
output = prediction_layer(x)

my_model = Model(base_model.input, output)
my_model.summary()

total_epoch = 10
initial_learning_rate = 0.00001
decay_rate = 0.96
decay_steps = 10000

# Create a learning rate scheduler

def lr_scheduler(epoch):
    epoch += 1
   
    if epoch == 1:
        return initial_learning_rate
    
    
    elif epoch >= 2 and epoch <= 40:
        return (0.2 * epoch ** 3) * math.exp(-0.45 * epoch) * initial_learning_rate
    
    else:
        return initial_learning_rate
    

stage = [i for i in range(0,25)]
learning_rate = [lr_scheduler(x) for x in stage]
plt.plot(stage, learning_rate)
print(learning_rate)


scheduler = tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)
earlystopping = EarlyStopping(monitor='val_accuracy', patience=5,restore_best_weights=True);
reduceLR = ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.1,
    patience=1,
    verbose=1, 
    mode='auto'
)
my_model.compile(optimizer=tf.optimizers.Adam(learning_rate=initial_learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint_filepath = './Checkpoints/85ModelCNN'
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    verbose=1,
    mode='max',
    save_best_only=True)

# path = "firsttestweights.h5"
# my_model.load_weights(path)

# plot_model(my_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(np.argmax(y_train, axis=1)), y=np.argmax(y_train, axis=1))
class_weight_dict = dict(enumerate(class_weights))

initial_epochs = 10
start_time = time.time()
hist = my_model.fit(
    train_ds,
    validation_data=val_ds,
    callbacks=[model_checkpoint_callback],
    epochs=10,
    class_weight=class_weight_dict  # Pass class weights here
)
execution_time = (time.time() - start_time)/60.0
print("Training execution time (mins)",execution_time)


acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']

loss = hist.history['loss']
val_loss = hist.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([0.3,1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0.3,2.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

base_model.trainable = True

print("Number of layers in the base model: ", len(base_model.layers))

fine_tune_at = 464

for layer in base_model.layers[:fine_tune_at]:
  layer.trainable = False

my_model.compile(optimizer=tf.optimizers.Adam(learning_rate=initial_learning_rate/10), loss='categorical_crossentropy', metrics=['accuracy'])
my_model.summary()

reduceLRfine = ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.1,
    patience=2,
    verbose=1, 
    mode='auto'
) 

new_earlystop = EarlyStopping(monitor='val_accuracy', patience=3, verbose=1);
checkpoint = ModelCheckpoint('.Checkpoints/resNet152V2_NN.hdf5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
fine_tune_epochs = 100
total_epochs =  initial_epochs + fine_tune_epochs

start_time = time.time() 
history_fine =my_model.fit(
        train_ds,
        validation_data=val_ds,
        callbacks=[new_earlystop,checkpoint,reduceLRfine],
        epochs=total_epochs,
        initial_epoch=hist.epoch[-1],
        class_weight=class_weight_dict)

execution_time = (time.time() - start_time)/60.0
print("Training execution time (mins)",execution_time)

my_model.save('TESTING-MODEL-Enhanced.h5', overwrite=True) 
my_model.save_weights('TESTING-MODEL-WEIGHTS-Enhanced.h5', overwrite=True)
print("Saved model to disk")

acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0, 1])
plt.plot([initial_epochs,initial_epochs],
          plt.ylim(), label='Start Fine Tuning')
plt.plot([len(acc)-6, len(acc)-6], plt.ylim(),
         label='Early Stopped', linestyle='--', color='black')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.5])
plt.plot([initial_epochs-1,initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.plot([len(acc)-6, len(acc)-6], plt.ylim(),
         label='Early Stopped', linestyle='--', color='black')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.ylabel('Cross Entropy')
plt.show()


evaluation_results = my_model.evaluate(test_ds)

# Extract loss and accuracy
test_loss = evaluation_results[0]
test_accuracy = evaluation_results[1]

# Print test loss and accuracy
print("Test loss: ", test_loss)
print("Test accuracy: ", test_accuracy)

# Make predictions on test data
y_pred = my_model.predict(test_ds)

# Convert predictions to class labels
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)  # Assuming y_test is one-hot encoded

# Compute F1-score
f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')

results_df = pd.DataFrame({'Metric': ['Test Loss', 'Test Accuracy', 'F1-score'],
                           'Value': [test_loss, test_accuracy, f1]})

results_df

import seaborn as sns

# Compute accuracy and F1-score
acc = accuracy_score(y_true_labels, y_pred_labels)
score = f1_score(y_true_labels, y_pred_labels, average='micro')

# Compute confusion matrix
cm = confusion_matrix(y_true_labels, y_pred_labels)
def_cm = pd.DataFrame(cm, index=breed_list,columns=breed_list)

# Plot confusion matrix
plt.figure(figsize=(15, 12))
sns.heatmap(def_cm, annot=True, annot_kws={'size': 8}, fmt='d')

# Rotate x-axis labels for better readability
plt.xticks(rotation=90)

plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title(f'Confusion Matrix\nAccuracy: {acc:.3f}; F1-score: {score:.3f}')
plt.show()

# Generate classification report as a string
# print

workbook = Workbook()
workbook.remove(workbook.active) # Delete default sheet.

report = (classification_report(y_true_labels, y_pred.argmax(1), target_names=breed_list,output_dict=True))

workbook = convert_report2excel(
    workbook=workbook,
    report=report,
    sheet_name="proposed_withweights_9010"
)

workbook.save("classification_report_proposedwithweights_9010.xlsx")

def plot_top_misclassified_pairs(y_true_labels, y_pred_labels, breed_list, top_n=30):
    # Create a dictionary to count misclassified pairs
    misclassified_pairs = {}

    # Iterate through each prediction and compare it with the true label
    for true_label, pred_label in zip(y_true_labels, y_pred_labels):
        true_breed = breed_list[true_label]
        pred_breed = breed_list[pred_label]

        # If the prediction is incorrect, increment the count for the misclassified pair
        if true_label != pred_label:
            pair = (true_breed, pred_breed)
            misclassified_pairs[pair] = misclassified_pairs.get(pair, 0) + 1

    # Sort the misclassified pairs by the number of occurrences in descending order
    sorted_pairs = sorted(misclassified_pairs.items(), key=lambda x: x[1], reverse=True)

    # Extract the top N misclassified pairs and their counts
    top_pairs = sorted_pairs[:top_n]

    # Extract breeds and counts for plotting
    breeds = ['{} / {}'.format(pair[0][0].split('-')[-1], pair[0][1].split('-')[-1]) for pair in top_pairs]
    counts = [pair[1] for pair in top_pairs]

    # Plot the top N misclassified pairs
    plt.figure(figsize=(15, 8))
    plt.barh(range(len(breeds)), counts, align='center')
    plt.yticks(range(len(breeds)), breeds)
    plt.xlabel('Number of Misclassifications')
    plt.ylabel('Misclassified Breeds')
    plt.title('Top {} Misclassified Breed Pairs'.format(top_n))
    plt.gca().invert_yaxis()  # Invert y-axis to display the highest count at the top
    plt.show()

# Call the function to plot the top 30 misclassified breed pairs
plot_top_misclassified_pairs(y_true_labels, y_pred_labels, breed_list)



