#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2

cap = cv2.VideoCapture(0)

stat = cap.isOpened()
print(stat)

while(True):
    ret, frame = cap.read()

    edges = cv2.Canny(frame, 10, 50)

    cv2.imshow('frame',frame)

    cv2.imshow('edges',edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
