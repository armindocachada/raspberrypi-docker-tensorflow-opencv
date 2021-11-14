from queue import Queue

import logging
import multiprocessing
import os
import sys
import traceback
from notifications.slack import *
from multiprocessing import Process, JoinableQueue
from enum import IntEnum
sys.path.append(os.path.dirname(os.getcwd()))


logger = logging.getLogger('wildlife_camera')

class NotificationType(IntEnum):
    FRAME = 1
    VIDEO = 2



class Notification(object):
    def __init__(self, type, item, labels=set(), incidentId=None):
        self.type = type
        self.item = item
        self.labels=labels
        self.imageUrl = None
        self.videoUrl = None
        self.incidentId = incidentId

class Notifications(multiprocessing.Process):


    def __init__(self, args):
        multiprocessing.Process.__init__(self)
        self.notifications = JoinableQueue()
        logger.info("called init thread for notifications")

        self.slack = Slack(args)


    def processNotification(self, item):
        logger.info("Processing notification for item:{}".format(item.type))

        try:
               print("Call Notify Slack")
               self.slack.notifySlack(item)
               print("After Notify Slack")
        except:
            logger.error("Unexpected error")
            traceback.print_exc()


    def notify(self, item):
        logger.info("Adding item to notification queue")
        self.notifications.put(item)


    def run(self):
        while True:
            logger.info("Looking for notifications to deliver from queue")
            item = self.notifications.get()
            if item is None:
                break
            self.processNotification(item)
            self.notifications.task_done()
