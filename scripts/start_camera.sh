#!/bin/bash

echo "$@"


DATA_DIR="/output"

cd ${DATA_DIR}

mkdir -p ${DATA_DIR}/tflite
mkdir -p ${DATA_DIR}/edgetpu

#
## Get TF Lite model and labels
curl -C - -O http://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
unzip -n coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip -d ${DATA_DIR}/tflite
rm coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
#
## Get a labels file with corrected indices, delete the other one
(cd ${DATA_DIR}/tflite && curl -C - -O  https://dl.google.com/coral/canned_models/coco_labels.txt)
rm ${DATA_DIR}/tflite/labelmap.txt
#
## Get version compiled for Edge TPU
(cd ${DATA_DIR}/edgetpu && curl -C - -O  https://dl.google.com/coral/canned_models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite)
## Get a labels file with corrected indices, delete the other one
(cd ${DATA_DIR}/edgetpu && curl -C - -O  https://dl.google.com/coral/canned_models/coco_labels.txt)
#
#
echo -e "Files downloaded to ${DATA_DIR}/"

cd /app/tflite && python3 wildlife_camera.py $@
