import logging
import platform
import sys

from threading import Timer

import greengrasssdk

from lne_tflite import interpreter as lt
import cv2
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

client = greengrasssdk.client("iot-data")

def resize_img(img, width, height, mode):
	fitted_img = cv2.resize(img, (width, height))
	fitted_img = cv2.cvtColor(fitted_img, cv2.COLOR_BGR2RGB)
	fitted_img = np.expand_dims(fitted_img, axis = 0)
	if mode == 1:
		fitted_img = fitted_img.astype('float32')
		fitted_img = (fitted_img -127.5) * (1.0 / 127.5)
	return fitted_img

def crop_image(img):
	(y,x,channel) = img.shape
	x_prime = y 
	img = img[0:y, int((x-x_prime)/2):int((x+x_prime)/2)]
	return img 

def camera_convert(frame):
	return np.transpose(frame.reshape(3, 480, 640), (1, 2, 0))

def greengrass_mobilenet_run():
	try:
		label = []

		interpreter = lt.Interpreter('./img_cls.lne')
		interpreter.allocate_tensors()

		with open('./labels.txt') as f:
			l_lines = f.readlines()
			labels = [ line for line in l_lines ]

		capture = cv2.VideoCapture(0)
		ret, frame = capture.read()
		img = camera_convert(frame)
		img = crop_image(img)
		input_data = resize_img(img, 128, 128, 1)

		input_details = interpreter.get_input_details()
		output_details = interpreter.get_output_details()

		interpreter.set_tensor(input_details[0]['index'], input_data)
		interpreter.invoke()
		output_data = interpreter.get_tensor(output_details[0]['index'])

		lne_answer = np.argmax(output_data)

		client.publish(topic="lge/img_clstest", queueFullPolicy="AllOrException", \
				payload="Result: {}".format(labels[lne_answer]))

	except Exception as e:
		logger.error("Failed to publish message: " + repr(e))

	Timer(5, greengrass_mobilenet_run).start()


greengrass_mobilenet_run()

def function_handler(event, context):
	return
