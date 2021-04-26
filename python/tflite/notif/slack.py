

import configparser
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from slack import WebClient
import io
import os
import cv2
from collections import defaultdict
import imutils
import logging
import threading
import sys
from enum import IntEnum
from notif.notifications import *
from myutils.videoutils import VideoUtils



logger = logging.getLogger('wildlife_camera')

class NotificationType(IntEnum):
    FRAME = 1
    VIDEO = 2


class Slack(object):

    def __init__(self, args):
        logger.info("called init thread for slack")
        self.config = configparser.ConfigParser()
        credentialsPath = args.slack_credentials
        slackConfigPath = "{}".format(credentialsPath)
        self.config.read(slackConfigPath)
        print(slackConfigPath)

        self.slack_token = self.ConfigSectionMap("Slack")['secrettoken']
        self.channel_id = self.ConfigSectionMap("Slack")['channelid']
        self.sc = WebClient(token=self.slack_token)


    def ConfigSectionMap(self,section):
        dict1 = {}
        options = self.config.options(section)
        logger.debug(options)
        for option in options:
            try:
                dict1[option] = self.config.get(section, option)
                if dict1[option] == -1:
                    logger.debug("skip: %s" % option)
            except:
                logger.error("exception on %s!" % option)
                dict1[option] = None
        return dict1

    def clearFiles(self):
        result = self.sc.api_call("files.list",
                        channel=self.channel_id,
                        types="images",
                        count=1000
                        )
        logger.info(result)
        logger.debug("deleting {}".format(len(result["files"])))
        for file in result["files"]:
                print("Deleting file {}".format(file["id"]))
                deleteFileResult = self.sc.api_call("files.delete",
                             file=file["id"])
                print("result = {} Succeeded = {} ".format(deleteFileResult, deleteFileResult["ok"]))
                if not deleteFileResult["ok"]:
                    break



    def notifySlack(self, notification):
        logger.info("Processing slack notification type {}".format( notification.type))
        logger.info("Slack.notifySlack Current thread:{}".format(threading.current_thread().name))

        if notification.type == NotificationType.FRAME:

            # if notification.type  == NotificationType.FRAME:
            logger.info("Processing frame")
            frame = notification.item
            # imageRgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imageFilename = "/tmp/image_to_upload.png"
            VideoUtils.saveImageToDisk(frame, imageFilename)


            print(repr(NotificationType.FRAME ))
            if NotificationType.FRAME == notification.type:
                print("I am happy")

            self.sc.files_upload(
                channels=self.channel_id,
                title='Object Detected',
                #initial_comment='Object Detected',
                file=imageFilename
            )



        elif notification.type == NotificationType.VIDEO:
            file = notification.item
            logger.info("Slack.sendVideo Current thread:{}".format(threading.current_thread().name))

            # is it mp4?
            fileNameMp4 = file
            if not file.endswith(".mp4"):
               fileNameMp4 = VideoUtils.convertToMp4(file)
               os.remove(file)

            result = self.sc.files_upload(
                 channels=self.channel_id,
                 title='Video captured',
                 # initial_comment='Object Detected',
                 file=fileNameMp4
             )

            logger.info("Upload video result={}".format(result["ok"]))





