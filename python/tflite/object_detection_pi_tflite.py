# Lint as: python3
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
# @author Armindo Cachada
# Description
#
#
#
# Based on https://github.com/google-coral/tflite/blob/master/python/examples/detection/detect_image.py
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import random
import picamera
import datetime
import numpy as np
import cv2
import time
import argparse
from picamera.array import PiRGBArray
from picamera import PiCamera
import logging
import detect
from imutils.video import VideoStream, FPS
import os
import tflite_runtime.interpreter as tflite

from PIL import Image
from PIL import ImageDraw

EDGETPU_SHARED_LIB="libedgetpu.so.1"

def setupLogger():
    # create logger with 'spam_application'
    myLogger = logging.getLogger('wildlife_camera')
    myLogger.setLevel(logging.INFO)
    # create file handler which logs even debug messages
    fh = logging.FileHandler('/output/wildlife_camera.log')
    fh.setLevel(logging.INFO)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    myLogger.addHandler(fh)
    myLogger.addHandler(ch)
    return myLogger


def findFileWithExtension(extension,dir_path):
    labelFile = None
    for f in os.listdir(dir_path):
        # List files with .py
        if f.endswith(extension):
            labelFile = os.path.join(dir_path, f)
            break
    return labelFile
def load_labels(modelDir, encoding='utf-8'):

  """Loads labels from file (with or without index numbers).
  Args:
    modelDir: path to directory containing model.
    encoding: label file encoding.
  Returns:
    Dictionary mapping indices to labels.
  """
  label_file = findFileWithExtension(".txt", modelDir)
  with open(label_file, 'r', encoding=encoding) as f:
    lines = f.readlines()
    if not lines:
      return {}

    if lines[0].split(' ', maxsplit=1)[0].isdigit():
      pairs = [line.split(' ', maxsplit=1) for line in lines]
      return {int(index): label.strip() for index, label in pairs}
    else:
      return {index: line.strip() for index, line in enumerate(lines)}


def make_interpreter(model_dir, enable_tpu):
  model_file = findFileWithExtension(".tflite", model_dir)
  model_file, *device = model_file.split('@')
  print(f"device={device}")
  # is edgetpu enabled
  if enable_tpu:
      return tflite.Interpreter(
          model_path=model_file,
          experimental_delegates=[
              tflite.load_delegate(EDGETPU_SHARED_LIB,
                                   {'device': device[0]} if device else {})
          ])
  else:
      return tflite.Interpreter(
          model_path=model_file
      )

# checks if an object of the given label has been detected
# and returns True if that is the case
def detected_object(objs, labelsToDetectStr, labels):
    labelsToDetect = labelsToDetectStr.split(",")
    for obj in objs:
        labelDetected = labels.get(obj.id, obj.id)
        if labelDetected in labelsToDetect:
            return True

    return False

def log_detected_objects(objs, objects_to_detect, labels):
    for obj in objs:
        bbox = obj.bbox
        # check if this object is in the list objects to detect - and set different color
        # red

        if detected_object([obj], objects_to_detect, labels):
            logger.info('Detected object= %s=%.2f' % (labels.get(obj.id, obj.id), obj.score))


def draw_objects(image, objs, objects_to_detect, labels):
  """Draws the bounding box and label for each object."""
  font = cv2.FONT_HERSHEY_SIMPLEX
  fontScale=1

  # Line thickness of 2 px
  thickness = 2

  for obj in objs:
    bbox = obj.bbox
    # check if this object is in the list objects to detect - and set different color
    # red
    detected_color = (0, 0, 255)
    if detected_object([obj], objects_to_detect, labels):
        # BGR - BLUE
        detected_color = (255, 0 , 0)

    cv2.rectangle(image,(bbox.xmax, bbox.ymax), (bbox.xmin, bbox.ymin),
                   detected_color)

    cv2.putText(image,  '%s=%.2f' % (labels.get(obj.id, obj.id), obj.score),
            (bbox.xmin + 10, bbox.ymin + 10),
            font,
            fontScale,
            detected_color,
            thickness,
            cv2.LINE_AA)

def enableCircularCameraRecording(piVideoStream):
    #enable circular stream
    camera = piVideoStream.camera
    stream = picamera.PiCameraCircularIO(camera, seconds=30)
    camera.start_recording(stream, format='h264')
    return stream

def recordVideoFromCamera(piVideoStream, circularStream):
    currentDate = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    piVideoStream.camera.wait_recording(10)
    circularStream.copy_to('/output/motion_' + currentDate + ".h264")
    #camera.stop_recording()

import urllib
import tarfile



def main():

  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)


  parser.add_argument('-t', '--threshold', type=float, default=0.5,
                      help='Score threshold for detected objects.')
  parser.add_argument('-c', '--count', type=int, default=5,
                      help='Number of times to run inference')
  parser.add_argument('-tpu', '--enable-tpu', action='store_true',
                      help='Whether TPU is enabled or not')
  parser.add_argument('-objects', '--detect-objects', type=str, default="bird")
  parser.add_argument('-debug', '--enable-debug', action='store_true',
                      help='Whether Debug is enabled or not - Webcamera viewed is displayed when in this mode')

  args = parser.parse_args()

  objects_to_detect = args.detect_objects

  if args.enable_tpu:
      model_dir = "/output/edgetpu/"
  else:
      "model_dir = "/output/tflite/"

  labels = load_labels(args.model)
  interpreter = make_interpreter(args.model, args.enable_tpu)
  interpreter.allocate_tensors()
  # begin detect video

  # initialize the camera and grab a reference to the raw camera capture
  # camera = PiCamera()
  resolution = (1280, 720)
  # camera.resolution = resolution
  # camera.framerate = 30

  freq = cv2.getTickFrequency()
  # rawCapture = PiRGBArray(camera, size=resolution)

  fps = FPS().start()
  piVideoStream = VideoStream(usePiCamera=True, resolution=resolution, framerate=30).start()

  # enable circular stream
  cameraCircularStream = enableCircularCameraRecording(piVideoStream)

  time.sleep(1)
  while True:
      t0 = cv2.getTickCount()

      frame = piVideoStream.read()
      fps.update()

      image_rgb_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

      # when passing the size to the method we reverse the tuple
      scale = detect.set_input(interpreter, image_rgb_np.shape[:2][::-1],
                               lambda size: cv2.resize(image_rgb_np,size, interpolation=cv2.INTER_AREA))
      interpreter.invoke()
      objs = detect.get_output(interpreter, args.threshold, scale)

      # we only draw bounding boxes and detection labels in the
      # frame if we are in debug mode
      if objs and args.enable_debug:
        draw_objects(frame, objs, objects_to_detect, labels)

      # we only record to video file if not in debug mode
      if  not args.enable_debug and detected_object(objs, objects_to_detect, labels):
        log_detected_objects(objs, objects_to_detect, labels)
        # record 20 s video clip - it will freeze main thread
        recordVideoFromCamera(piVideoStream,cameraCircularStream)
      # in debug mode we show the object detection boxes


      if args.enable_debug:
          cv2.imshow('frame', frame)

      if cv2.waitKey(1) & 0xFF == ord('q'):
          break
  fps.stop()
  piVideoStream.stop()

  print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
  print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

  cv2.destroyAllWindows()



if __name__ == '__main__':
  logger = setupLogger()
  main()

































