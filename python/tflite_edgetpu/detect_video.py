# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Example using TF Lite to detect objects in a given image."""

import argparse
import time
import cv2
from PIL import Image
from PIL import ImageDraw

import detect
import tflite_runtime.interpreter as tflite
import platform
from imutils.video import VideoStream, FPS
EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]


def load_labels(path, encoding='utf-8'):
  """Loads labels from file (with or without index numbers).
  Args:
    path: path to label file.
    encoding: label file encoding.
  Returns:
    Dictionary mapping indices to labels.
  """
  with open(path, 'r', encoding=encoding) as f:
    lines = f.readlines()
    if not lines:
      return {}

    if lines[0].split(' ', maxsplit=1)[0].isdigit():
      pairs = [line.split(' ', maxsplit=1) for line in lines]
      return {int(index): label.strip() for index, label in pairs}
    else:
      return {index: line.strip() for index, line in enumerate(lines)}


def make_interpreter(model_file):
  model_file, *device = model_file.split('@')
  return tflite.Interpreter(
      model_path=model_file,
      experimental_delegates=[
          tflite.load_delegate(EDGETPU_SHARED_LIB,
                               {'device': device[0]} if device else {})
      ])


def draw_objects(frame, objs, labels):
  """Draws the bounding box and label for each object."""
  for obj in objs:
    bbox = obj.bbox
    cv2.rectangle(frame, (bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax), (0,0,255))
    cv2.putText(frame,
               '%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),
                (bbox.xmin + 10, bbox.ymin + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                     (0, 0, 255), 2, cv2.LINE_AA)


def main():
  print("Yo")
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('-m', '--model', required=True,
                      help='File path of .tflite file.')
  parser.add_argument('-l', '--labels',
                      help='File path of labels file.')
  parser.add_argument('-t', '--threshold', type=float, default=0.4,
                      help='Score threshold for detected objects.')
  parser.add_argument('-o', '--output',
                      help='File path for the result image with annotations')
  parser.add_argument('-c', '--count', type=int, default=5,
                      help='Number of times to run inference')
  args = parser.parse_args()

  labels = load_labels(args.labels) if args.labels else {}
  interpreter = make_interpreter(args.model)
  interpreter.allocate_tensors()

  # initialize the camera and grab a reference to the raw camera capture
  # camera = PiCamera()
  resolution = (1280, 720)
  # camera.resolution = resolution
  # camera.framerate = 30

  freq = cv2.getTickFrequency()
  # rawCapture = PiRGBArray(camera, size=resolution)

  fps = FPS().start()
  piVideoStream = VideoStream(usePiCamera=True, resolution=resolution, framerate=30).start()

  time.sleep(1)
  while True:
      t0 = cv2.getTickCount()

      frame = piVideoStream.read()
      fps.update()

      image_rgb_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

      #    image_rgb_np_with_detections = doinference(image_rgb_np)

      #    image_bgr_np_with_detections = cv2.cvtColor(image_rgb_np_with_detections, cv2.COLOR_RGB2BGR)
      #     cv2.putText(frame, 'FPS: {0:.2f}'.format(fps.fps()), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
      #                 (255, 255, 0), 2, cv2.LINE_AA)
      #

      scale = detect.set_input(interpreter, resolution,
                               lambda size: cv2.resize(image_rgb_np, size))

      interpreter.invoke()
      objs = detect.get_output(interpreter, args.threshold, scale)

      draw_objects(frame, objs, labels)

      # for obj in objs:
      #     print(labels.get(obj.id, obj.id))
      #     print('  id:    ', obj.id)
      #     print('  score: ', obj.score)
      #     print('  bbox:  ', obj.bbox)

      cv2.imshow('frame', frame)
      #
      #     t1 = cv2.getTickCount()
      #     time= (t1-t0)/freq
      #     fps = 1/time
      #     # clear the stream in preparation for the next frame
      #     rawCapture.truncate(0)
      #     # resets the time

      if cv2.waitKey(1) & 0xFF == ord('q'):
          break
  fps.stop()
  piVideoStream.stop()

  print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
  print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

  cv2.destroyAllWindows()





if __name__ == '__main__':
 main()