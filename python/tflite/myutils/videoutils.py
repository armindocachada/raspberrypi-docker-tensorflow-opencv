import datetime
import re
import os
import subprocess as sp
import json
import cv2
import glob
import logging


logger = logging.getLogger('wildlife_camera')

class VideoUtils(object):


    @classmethod
    def convertToMp4(cls, filePath, framerate='20'):
        FFMPEG_BIN = "/usr/bin/ffmpeg"
        command = [FFMPEG_BIN,
                   '-hide_banner',
                   '-loglevel', 'panic',
                   '-i', filePath,
                   '-c:v', 'copy',
                   '-r', framerate,
                   filePath + ".mp4"
                   ]
        sp.check_call(command)
        logger.info(f"Converting file {filePath} to .mp4 using ffmpeg")
        return filePath + ".mp4"


    @classmethod
    def saveImageToDisk(cls, frame, path):
        # save frame to disk, to allow
        # easy configuration of security zones
        cv2.imwrite(path, frame)
        # cv2.imshow('frame', frame)
