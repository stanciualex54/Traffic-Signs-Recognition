from tensorflow.keras.models import load_model
from skimage import transform
from skimage import exposure
from skimage import io
from imutils import paths
import numpy as np
import argparse
import imutils
import random
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="calea către modelul pre-antrenat")
ap.add_argument("-i", "--images", required=True,
	help="calea către directorul care conține imaginile de test")
ap.add_argument("-e", "--examples", required=True,
	help="calea către directorul care conține imaginile predicționate")
args = vars(ap.parse_args())

print("[INFO] încărcare model...")
model = load_model(args["model"])

labelNames = open("venv\labelsro.csv").read().strip().split("\n")[1:]
labelNames = [l.split(",")[1] for l in labelNames]

print("[INFO] se realizează predicția...")
imagePaths = list(paths.list_images(args["images"]))
random.shuffle(imagePaths)
imagePaths = imagePaths[:50]

for (i, imagePath) in enumerate(imagePaths):
	image = io.imread(imagePath)
	image = transform.resize(image, (32, 32))
	image = exposure.equalize_adapthist(image, clip_limit=0.1)
	image = image / 255.0
	image = np.expand_dims(image, axis=0)

	predictions = model.predict(image)
	j = predictions.argmax(axis=1)[0]
	label = labelNames[j]

	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=175)
	cv2.putText(image, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
		0.40, (0, 0, 255), 2)

	p = os.path.sep.join([args["examples"], "{}.png".format(i)])
	cv2.imwrite(p, image)