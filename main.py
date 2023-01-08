# import the necessary packages
import tensorflow
import keras
from keras.applications.vgg16 import preprocess_input
from keras.applications import ResNet50
from keras.layers import Flatten
from keras.layers import Dense,BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from keras.layers import Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.models import Sequential
from keras.utils import img_to_array
from keras.utils import load_img
from keras.metrics import Accuracy, FalseNegatives, FalsePositives, TrueNegatives,TruePositives, Precision, Recall, AUC, BinaryAccuracy
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import numpy as np
from numpy import expand_dims
import warnings
warnings.filterwarnings('ignore')
import cv2
import os
import pickle


# define the base path to the input dataset and then use it to derive
# the path to the images directory and annotation CSV file
BASE_PATH = "dataset"
Train_IMAGES_PATH = os.path.sep.join([BASE_PATH, "train_images"])
Train_ANNOTS_PATH = os.path.sep.join([BASE_PATH, "train_annotations.csv"])
Validation_IMAGES_PATH = os.path.sep.join([BASE_PATH, "ValidationImages"])
Validation_ANNOTS_PATH = os.path.sep.join([BASE_PATH, "validation_annotations.csv"])


# define the path to the base output directory
BASE_OUTPUT = "output"

# define the path to the output serialized model, model training plot,
# and testing image filenames
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "detector.h5"])
LOSS_PATH = os.path.sep.join([BASE_OUTPUT, "loss.png"])
ACCURACY_PATH = os.path.sep.join([BASE_OUTPUT, "acc.png"])
FEATUREMAP_PATH = os.path.sep.join([BASE_OUTPUT, "featuremap.png"])
# TEST_FILENAMES = os.path.sep.join([BASE_OUTPUT, "test_images.txt"])
LB_PATH = os.path.sep.join([BASE_OUTPUT, "lb.pickle"])


# initialize our initial learning rate, number of epochs to train
# for, and the batch size
INIT_LR = 1e-4
NUM_EPOCHS = 50
BATCH_SIZE = 32


# define the names of the classes
CLASSES = ["Human", "No Human"]

# load the contents of the CSV annotations file
# print("[INFO] loading dataset...")
# rows = open(ANNOTS_PATH).read().strip().split("\n")
print("[INFO] loading dataset...")
train_rows = open(Train_ANNOTS_PATH).read().strip().split("\n")
validation_rows = open(Validation_ANNOTS_PATH).read().strip().split("\n")

# initialize the list of data (images), our target output predictions
# (bounding box coordinates), along with the filenames of the
# individual images
train_data = []
trainFilenames = []
train_imagePaths = []
validation_data = []
validationFilenames = []
validation_imagePaths = []
labels = []
labels_validation= []
trainbboxes = []
validationbboxes = []

# loop over all CSV files in the annotations directory
# loop over the rows
# for row in rows:
# 	# break the row into the filename, bounding box coordinates,
# 	# and class label
# 	row = row.split(",")
# 	(name, width, height, label, xtl, ytl, xbr, ybr, z_order) = row
# 	# derive the path to the input image, load the image (in OpenCV
# 	# format), and grab its dimensions
# 	imagePath = os.path.sep.join([IMAGES_PATH, name])
# 	#print(imagePath)
# 	image = cv2.imread(imagePath)
# 	# plt.imshow(image)
# 	# plt.show()
# 	(h, w) = image.shape[:2]
#
# 	# scale the bounding box coordinates relative to the spatial
# 	# dimensions of the input image
# 	if xtl != "":
# 		xtl = float(xtl) / w
# 		ytl = float(ytl) / h
# 		xbr = float(xbr) / w
# 		ybr = float(ybr) / h
# 	else:
# 		xtl = 0
# 		ytl = 0
# 		xbr = 0
# 		ybr = 0
# 		label = "No Human"
# 	# load the image and preprocess it
# 	image = load_img(imagePath, target_size=(224, 224))
# 	image = img_to_array(image)
# 	# update our list of data, targets, and filenames
# 	data.append(image)
# 	labels.append(label)
# 	bboxes.append((xtl, ytl, xbr, ybr))
# 	filenames.append(name)
# 	imagePaths.append(imagePath)
# # convert the data and targets to NumPy arrays, scaling the input
# # pixel intensities from the range [0, 255] to [0, 1]
# data = np.array(data, dtype="float32")/ 255.0
# labels = np.array(labels)
# bboxes = np.array(bboxes, dtype="float32")
# imagePaths = np.array(imagePaths)
#
# #print(data)
#
# # partition the data into training and testing splits using 90% of
# # the data for training and the remaining 10% for testing
# split = train_test_split(data, bboxes, filenames, test_size=0.10,
# 	random_state=42)
# # perform one-hot encoding on the labels
# lb = LabelBinarizer()
# labels = lb.fit_transform(labels)
# # only there are only two labels in the dataset, then we need to use
# # Keras/TensorFlow's utility function as well
# if len(lb.classes_) == 2:
# 	labels = to_categorical(labels)
# # unpack the data split
# (trainImages, testImages) = split[:2]
# (trainBBoxes, testBBoxes) = split[2:4]
# (trainFilenames, testFilenames) = split[4:]
#
#
# # write the testing filenames to disk so that we can use then
# # when evaluating/testing our bounding box regressor
# print("[INFO] saving testing filenames...")
# f = open(TEST_FILENAMES, "w")
# print(TEST_FILENAMES)
# f.write("\n".join(TEST_FILENAMES))
# f.close()
for train_row in train_rows:
	# break the row into the filename, bounding box coordinates,
	# and class label
	train_row = train_row.split(",")
	(name, width, height, label, xtl, ytl, xbr, ybr, z_order) = train_row
	# derive the path to the input image, load the image (in OpenCV
	# format), and grab its dimensions
	train_imagePath = os.path.sep.join([Train_IMAGES_PATH, name])
	#print(imagePath)
	image = cv2.imread(train_imagePath)
	# plt.imshow(image)
	# plt.show()
	(h, w) = image.shape[:2]
	# scale the bounding box coordinates relative to the spatial
	# dimensions of the input image
	if xtl != "":
		xtl = float(xtl) / w
		ytl = float(ytl) / h
		xbr = float(xbr) / w
		ybr = float(ybr) / h
	else:
		xtl = 0
		ytl = 0
		xbr = 0
		ybr = 0
		label = "No Human"
	# load the image and preprocess it
	train_image = load_img(train_imagePath, target_size=(224, 224, 3))
	train_image = img_to_array(train_image)
	# update our list of data, targets, and filenames
	train_data.append(train_image)
	labels.append(label)
	trainbboxes.append((xtl, ytl, xbr, ybr))
	trainFilenames.append(train_image)
	train_imagePaths.append(train_imagePath)
# convert the data and targets to NumPy arrays, scaling the input
# pixel intensities from the range [0, 255] to [0, 1]
train_data = np.array(train_data, dtype="float32") / 255.0
# if xtl != "":
# 	bboxes = np.array(bboxes, dtype="float32")
train_labels = np.array(labels)
train_boxes = np.array(trainbboxes, dtype="float32")
print(len(train_boxes))
print(len(train_data))
train_imagePaths = np.array(train_imagePath)
# print(data)
for validation_row in validation_rows:
	# break the row into the filename, bounding box coordinates,
	# and class label
	validation_row = validation_row.split(",")
	(name, width, height, label, xtl, ytl, xbr, ybr, z_order) = validation_row
	# derive the path to the input image, load the image (in OpenCV
	# format), and grab its dimensions
	validation_imagePath = os.path.sep.join([Validation_IMAGES_PATH, name])
	# print(imagePath)
	image = cv2.imread(validation_imagePath)
	# plt.imshow(image)
	# plt.show()
	(h, w) = image.shape[:2]
	# scale the bounding box coordinates relative to the spatial
	# dimensions of the input image
	if xtl != "":
		xtl = float(xtl) / w
		ytl = float(ytl) / h
		xbr = float(xbr) / w
		ybr = float(ybr) / h
	else:
		xtl = 0
		ytl = 0
		xbr = 0
		ybr = 0
		label = "No Human"
	# load the image and preprocess it
	validation_image = load_img(validation_imagePath, target_size=(224, 224, 3))
	validation_image = img_to_array(validation_image)
	# update our list of data, targets, and filenames
	validation_data.append(validation_image)
	labels_validation.append(label)
	validationbboxes.append((xtl, ytl, xbr, ybr))
	validationFilenames.append(validation_image)
	validation_imagePaths.append(validation_imagePath)
# convert the data and targets to NumPy arrays, scaling the input
# pixel intensities from the range [0, 255] to [0, 1]
validation_data = np.array(validation_data, dtype="float32") / 255.0
# if xtl != "":
# 	bboxes = np.array(bboxes, dtype="float32")
validation_labels = np.array(labels_validation)
validation_boxes = np.array(validationbboxes, dtype="float32")
train_imagePaths = np.array(validation_imagePath)
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# only there are only two labels in the dataset, then we need to use
# Keras/TensorFlow's utility function as well
if len(lb.classes_) == 2:
	labels = to_categorical(labels)
# determine the total number of image paths in training, validation,
# and testing directories
# totalTrain = len(trainImages)
# totalVal = len(testImages)
# print(totalTrain,totalVal)
# load the resnet network, ensuring the head FC layers are left off
print("[INFO] preparing model...")
# construct the head of the model that will be placed on top of the
# the base model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224, 3))
base_model.trainable = False
chopped_resnet = Model(inputs=[base_model.input], outputs=[base_model.layers[90].output])
classification_output = GlobalAveragePooling2D()(chopped_resnet.output)
# classification_output = Flatten()(chopped_resnet.output)
classification_output = Dense(units=1, activation='sigmoid')(classification_output)
localization_output = Flatten()(chopped_resnet.output)
localization_output = Dense(units=4, activation='relu')(localization_output)
model = Model(inputs=[chopped_resnet.input], outputs=[classification_output, localization_output])
model.summary()
# place the head FC model on top of the base model (this will become
# the actual model we will train)
# model = Model(inputs=baseModel.input, outputs=headModel)
# # loop over all layers in the base model and freeze them so they will
# # *not* be updated during the training process
# for layer in baseModel.layers:
# 	layer.trainable = False
early_stopping = keras.callbacks.EarlyStopping(monitor='loss', patience=5,mode='auto')
callback = keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

# compile the model

model.compile(optimizer='adam', metrics=['accuracy'],loss=['binary_crossentropy', 'mse'],loss_weights=[800, 1] )
print(model.summary())
# train the model
print("[INFO] training model...")
H = model.fit(train_data, [train_boxes[:, 0],train_boxes[:, 1]], batch_size=32, epochs=50, verbose=1,
 	validation_data=(validation_data, [validation_boxes[:, 0],validation_boxes[:, 1]]),
    shuffle=True,callbacks=[callback,early_stopping])
stopped_epoch = (early_stopping.stopped_epoch+1)
print(stopped_epoch)
# serialize the model to disk
print("[INFO] saving object detector model...")
model.save(MODEL_PATH, save_format="h5")
# serialize the model to disk
print("[INFO] saving object detector model...")
model.save(MODEL_PATH, save_format="h5")
for i in range(len(model.layers)):
	layer = model.layers[i]
	if 'conv' not in layer.name:
		continue
	print(i, layer.name, layer.output.shape)
model = Model(inputs=model.inputs, outputs=model.layers[1].output)
# expand dimensions so that it represents a single 'sample'
image = expand_dims(train_image, axis=0)
# prepare the image (e.g. scale pixel values for the vgg)
image = preprocess_input(image)
# get feature map for first hidden layer
feature_maps = model.predict(image)
fig = plt.figure(figsize=(20, 15))
for i in range(1, feature_maps.shape[3] + 1):
	plt.subplot(8, 8, i)
	plt.imshow(feature_maps[0, :, :, i - 1], cmap='gray')
plt.savefig(FEATUREMAP_PATH)
plt.show()
# serialize the label binarizer to disk
print("[INFO] saving label binarizer...")
f = open(LB_PATH, "wb")
f.write(pickle.dumps(lb))
f.close()
# plot the total loss, label loss, and bounding box loss
lossNames = ["loss"]
N = np.arange(0, NUM_EPOCHS)
plt.style.use("ggplot")
(fig, ax) = plt.subplots(3, 1, figsize=(13, 13))

# create a new figure for the accuracies
N = stopped_epoch
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, stopped_epoch), H.history["dense_loss"], label="train_loss")
plt.plot(np.arange(0, stopped_epoch), H.history["val_dense_loss"], label="val_loss")
plt.title("Bounding Box Regression Loss on Training Set")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(LOSS_PATH)
plt.close()
# #create a new figure for the accuracies
N = stopped_epoch
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, stopped_epoch), H.history["dense_accuracy"], label="Accuracy")
plt.plot(np.arange(0, stopped_epoch), H.history["val_dense_accuracy"], label="val_Accuracy")
plt.title("Bounding Box Regression Loss on Training Set")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")
plt.savefig(ACCURACY_PATH)
plt.close()



