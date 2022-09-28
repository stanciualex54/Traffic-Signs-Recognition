import matplotlib
matplotlib.use("Agg")
from modelScript.trafficSigns import TrafficSignNetwork
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from skimage import transform
from skimage import exposure
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import os

def load_split(basePath, csvPath):

	data = []
	labels = []
	rows = open(csvPath).read().strip().split("\n")[1:]
	random.shuffle(rows)

	for (i, row) in enumerate(rows):
		if i > 0 and i % 1000 == 0:
			print("[INFO] s-au procesat {} de imagini".format(i))
		(label, imagePath) = row.strip().split(",")[-2:]
		imagePath = os.path.sep.join([basePath, imagePath])
		image = io.imread(imagePath)

		image = transform.resize(image, (32, 32))
		image = exposure.equalize_adapthist(image, clip_limit=0.1)
		data.append(image)
		labels.append(int(label))
	data = np.array(data)
	labels = np.array(labels)
	return (data, labels)

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="calea către setul de date GTSRB")
ap.add_argument("-m", "--model", required=True,
	help="calea către modelul antrenat")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="calea către graficul cu istoricul antrenamentului")
args = vars(ap.parse_args())

epochs = 20
learning_rate = 1e-3
batch_size = 32

labelNames = open("venv\labelsro.csv").read().strip().split("\n")[1:]
labelNames = [l.split(",")[1] for l in labelNames]

trainPath = os.path.sep.join([args["dataset"], "Train.csv"])
testPath = os.path.sep.join([args["dataset"], "Test.csv"])

print("[INFO] se încarcă datele de antrenament și test...")
(trainX, trainY) = load_split(args["dataset"], trainPath)
(testX, testY) = load_split(args["dataset"], testPath)

trainX = trainX / 255.0
testX = testX / 255.0

numberOfLabels = len(np.unique(trainY))
trainY = to_categorical(trainY, numberOfLabels)
testY = to_categorical(testY, numberOfLabels)

classTotals = trainY.sum(axis=0)
classWeight = dict()

for i in range(0, len(classTotals)):
	classWeight[i] = classTotals.max() / classTotals[i]

aug = ImageDataGenerator(
	rotation_range=10,
	zoom_range=0.15,
	width_shift_range=0.1,
	height_shift_range=0.1,
	shear_range=0.15,
	horizontal_flip=False,
	vertical_flip=False,
	fill_mode="nearest")

print("[INFO] compilare model...")
opt = Adam(learning_rate=learning_rate, decay=learning_rate / (epochs * 0.5))

model = TrafficSignNetwork.build(width=32, height=32, depth=3,
	classes=numberOfLabels)

model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

print("[INFO] antrenare rețea...")
History = model.fit(
	aug.flow(trainX, trainY, batch_size=batch_size),
	validation_data=(testX, testY),
	steps_per_epoch=trainX.shape[0] // batch_size,
	epochs=epochs,
	class_weight=classWeight,
	verbose=1)

print("[INFO] evaluare rețea...")
predictions = model.predict(testX, batch_size=batch_size)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=labelNames))

model.save(args["model"])


# plot the training loss and accuracy
N = np.arange(0, epochs)
plt.style.use("ggplot")
plt.figure(0)
plt.plot(N, History.history["loss"], label="train_loss")
plt.plot(N, History.history["val_loss"], label="val_loss")
plt.plot(N, History.history["accuracy"], label="train_acc")
plt.plot(N, History.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")
plt.savefig(args["plot"])
