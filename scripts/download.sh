#!/bin/bash

if [ $# -eq 0 ]; then
  DATA_DIR="/tmp"
else
  DATA_DIR="$1"
fi

cd ${DATA_DIR}

mkdir -p ${DATA_DIR}/tflite
mkdir -p ${DATA_DIR}/edgetpu


# Get TF Lite model and labels
curl -C - -O http://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
unzip -n coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip -d ${DATA_DIR}/tflite
rm coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip

# Get a labels file with corrected indices, delete the other one
(cd ${DATA_DIR}/tflite && curl -C - -O  https://dl.google.com/coral/canned_models/coco_labels.txt)
rm ${DATA_DIR}/tflite/labelmap.txt

# Get version compiled for Edge TPU
(cd ${DATA_DIR}/edgetpu && curl -C - -O  https://dl.google.com/coral/canned_models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite)
# Get a labels file with corrected indices, delete the other one
(cd ${DATA_DIR}/edgetpu && curl -C - -O  https://dl.google.com/coral/canned_models/coco_labels.txt)


echo -e "Files downloaded to ${DATA_DIR}/"

# with edgetpu
# cd /app/ && python3 tflite_edgetpu/object_detection_pi_tflite.py -m /tmp/edgetpu/ --enable-tpu -objects person

# without edgetpu