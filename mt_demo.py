from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
from darknet import Darknet
from preprocess import prep_image, inp_to_image, letterbox_image
import random
import pickle as pkl
import argparse
import pandas as pd
import accuracy_ex_torch
import glob
import os

def get_test_input(input_dim, CUDA):
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (input_dim, input_dim))
    img_ = img[:, :, ::-1].transpose((2, 0, 1))
    img_ = img_[np.newaxis, :, :, :] / 255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)

    if CUDA:
        img_ = img_.cuda()

    return img_


def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

def write(x, img, classes, colors, frames):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())

    cls = int(x[-1])
    label = "{0}".format(classes[cls])

    # print('frame: {}, x0:{}, x1:{}, y0:{}, y1:{}, class:{}'.format(frames, c1[0], c2[0], c1[1], c2[1], label))
    # print('c2:{}'.format(c2))
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    # print('c1:{}, c2:{}'.format(c1, c2))
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1);
    return img

def model_init():
    CUDA = torch.cuda.is_available()

    CUDA = torch.cuda.is_available()

    print("Loading network.....")
    model = Darknet("cfg/yolov3.cfg")
    model.load_weights("yolov3.weights")
    print("Network successfully loaded")

    if CUDA:
        model.cuda()

    return model, CUDA

class YOLO(object):
    def __init__(self):
        self.confidence = float(0.5)

        self.nms_thesh = float(0.4)

        self.num_classes = 80

        self.bbox_attrs = 5 + self.num_classes

        self.model, self.CUDA = model_init()

        self.model.net_info["height"] = 320

        self.skip_flag = 0

        self.data = np.array([]).reshape((-1, 6))

    def model_switch(self, reso):
        self.model.net_info["height"] = reso

    def init(self):
        pre_time = time.time()

    def run(self, frame, frames):
        if self.skip_flag == 0:
            inp_dim = int(self.model.net_info["height"])
            assert inp_dim % 32 == 0
            assert inp_dim > 32

            self.model.eval()
            img, orig_im, dim = prep_image(frame, inp_dim)
            im_dim = torch.FloatTensor(dim).repeat(1, 2)

            if self.CUDA:
                im_dim = im_dim.cuda()
                img = img.cuda()

            with torch.no_grad():
                output = self.model(Variable(img), self.CUDA)

            output = write_results(output, self.confidence, self.num_classes, nms=True, nms_conf=self.nms_thesh)
            if type(output) == int:
                return frame

            im_dim = im_dim.repeat(output.size(0), 1)
            scaling_factor = torch.min(inp_dim / im_dim, 1)[0].view(-1, 1)

            output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim[:, 0].view(-1, 1)) / 2
            output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim[:, 1].view(-1, 1)) / 2

            output[:, 1:5] /= scaling_factor

            for i in range(output.shape[0]):
                output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
                output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])

            classes = load_classes('data/coco.names')
            colors = pkl.load(open("pallete", "rb"))

            # print('output: {}'.format(output.shape[0]))
            for i in range(output.shape[0]):
                data_list = np.array([[frames, int(output[i, 1]), int(output[i, 3]), int(output[i, 2]),
                                       int(output[i, 4]), classes[int(output[i, 7])]]])
                self.data = np.vstack([self.data, data_list])
                # print(self.data)

            # print(self.data)
            list(map(lambda x: write(x, orig_im, classes, colors, frames, ), output))
            return orig_im

    def run_1(self,frame, frames):
        if self.skip_flag == 0:
            inp_dim = int(self.model.net_info["height"])
            assert inp_dim % 32 == 0
            assert inp_dim > 32

            self.model.eval()
            img, orig_im, dim = prep_image(frame, inp_dim)
            im_dim = torch.FloatTensor(dim).repeat(1, 2)

            if self.CUDA:
                im_dim = im_dim.cuda()
                img = img.cuda()

            with torch.no_grad():
                output = self.model(Variable(img), self.CUDA)

            output = write_results(output, self.confidence, self.num_classes, nms=True, nms_conf=self.nms_thesh)
            if type(output) == int:
                return frame

            im_dim = im_dim.repeat(output.size(0), 1)
            scaling_factor = torch.min(inp_dim / im_dim, 1)[0].view(-1, 1)

            output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim[:, 0].view(-1, 1)) / 2
            output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim[:, 1].view(-1, 1)) / 2

            output[:, 1:5] /= scaling_factor

            for i in range(output.shape[0]):
                output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
                output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])

            return output

def bbox(func):
    outs = func()
    bbox =[]
    def wrapper():
        for out in outs:
            bbox.append(out)
            print(bbox)

def create_tracker_by_name(tracker_type):
    tracker_types = ['boosting', 'mil', 'kcf', 'tld', 'medianflow', 'goturn', 'mosse', 'csrt']
    # Create a tracker based on tracker name
    if tracker_type == tracker_types[0]:
        tracker = cv2.TrackerBoosting_create()
    elif tracker_type == tracker_types[1]:
        tracker = cv2.TrackerMIL_create()
    elif tracker_type == tracker_types[2]:
        tracker = cv2.TrackerKCF_create()
    elif tracker_type == tracker_types[3]:
        tracker = cv2.TrackerTLD_create()
    elif tracker_type == tracker_types[4]:
        tracker = cv2.TrackerMedianFlow_create()
    elif tracker_type == tracker_types[5]:
        tracker = cv2.TrackerGOTURN_create()
    elif tracker_type == tracker_types[6]:
        tracker = cv2.TrackerMOSSE_create()
    elif tracker_type == tracker_types[7]:
        tracker = cv2.TrackerCSRT_create()
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available trackers are:')
        for t in tracker_types:
            print(t)
    return tracker



def mt_test():
    yolo = YOLO()
    videofile = '8.mp4'

    tracker_type = 'csrt'
    multi_tracker =cv2.MultiTracker_creat()
    tracker = create_tracker_by_name(tracker_type)
    cap = cv2.VideoCapture(videofile)

    assert cap.isOpened(), 'Cannot capture source'

    selection = [320, 416, 608]
    frames = 0
    out_fps = 1 / 24

    start = time.time()
    selection = [320, 416, 608]
    while cap.isOpened():

        ret, frame = cap.read()
        # yolo.model_switch(selection[np.random.randint(3)])

        if ret:
            if frames == 0:
                print('initialization')
                pre_time = time.time()
                print('frame:{}'.format(frames))
                orig_im = yolo.run(frame, frames)
                cv2.imshow("frame", orig_im)
                yolo_time = time.time() - pre_time
                print('fps:{} ,run tim:{}'.format(1 / yolo_time, yolo_time))
                #print("FPS of the video is {:5.2f}".format(frames / (time.time() - start)))
                # key = cv2.waitKey(1)
                # if key & 0xFF == ord('q'):
                #     break

            else:
                cv2.imshow("frame", frame)

            key = cv2.waitKey(1000)
            if key & 0xFF == ord('q'):
                break

            frames += 1

        else:

            break

if __name__ == '__main__':
    mt_test()