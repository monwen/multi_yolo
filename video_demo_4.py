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


def arg_parse():
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(description='YOLO v3 Video Detection Module')

    parser.add_argument("--video", dest='video', help=
    "Video to run detection upon",
                        default="video.avi", type=str)
    parser.add_argument("--dataset", dest="dataset", help="Dataset on which the network has been trained",
                        default="pascal")
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument("--cfg", dest='cfgfile', help=
    "Config file",
                        default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help=
    "weightsfile",
                        default="yolov3.weights", type=str)
    parser.add_argument("--reso", dest='reso', help=
    "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="416", type=str)
    return parser.parse_args()


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


def model_switch(reso):
    pass


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


def f1_avg(f1_score, f1_list, n):
    frame_n = len(f1_list)
    # print(frame_n)
    # print(n)
    f1_avg = 0
    if frame_n == n:
        f1_list = np.append(f1_list, f1_score)
        f1_avg = np.average(f1_list)
        f1_list = np.array([])
        # print('check1')
    else:
        f1_list = np.append(f1_list, f1_score)
        # print('check2')
        # print('f1_list:{}'.format(f1_list))

    return f1_avg, f1_list


def fetch_base(videofile, savefile):
    yolo = YOLO()
    # videofile = 'Video1.mov'
    cap = cv2.VideoCapture(videofile)

    assert cap.isOpened(), 'Cannot capture source'

    # selection = [320, 416, 608]
    frames = 0
    start = time.time()
    while cap.isOpened():

        ret, frame = cap.read()
        if ret:
            # print('frame:{}'.format(frames))
            # reso = selection[np.random.randint(3)]
            # yolo.model_switch(reso)
            pre_time = time.time()
            orig_im = yolo.run(frame, frames)
            # print('reso:{}'.format(yolo.model.net_info["height"]))
            # cv2.imshow("frame", orig_im)

            frames += 1

            print("FPS of the video is {:5.2f}, run time: {:5.2f}".format(1 / (time.time() - pre_time),
                                                                          time.time() - pre_time))
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break


        else:
            base = pd.DataFrame(
                {'frame': yolo.data[:, 0], 'x0': yolo.data[:, 1], 'x1': yolo.data[:, 2], 'y0': yolo.data[:, 3],
                 'y1': yolo.data[:, 4], 'class': yolo.data[:, 5]})
            base.to_csv(savefile, encoding='utf-8', index=False)
            break


def acc_test():
    yolo = YOLO()
    videofile = 'Video1.mov'

    ## base_line initiation
    # acc = accuracy_ex_torch.ACCU()
    # acc.load('base_608_txt')
    ## end base line initiation

    cap = cv2.VideoCapture(videofile)

    assert cap.isOpened(), 'Cannot capture source'

    selection = [320, 416, 608]
    frames = 0
    start = time.time()
    while cap.isOpened():

        ret, frame = cap.read()
        if ret:
            pre_time = time.time()
            print('frame:{}'.format(frames))
            # reso = selection[np.random.randint(3)]
            # yolo.model_switch(reso)

            orig_im = yolo.run(frame, frames)
            # print('yolo data: {}'.format(yolo.data))
            f1 = acc.get_acc(yolo.data)
            # print('reso:{}'.format(yolo.model.net_info["height"]))
            cv2.imshow("frame", orig_im)

            frames += 1

            print("FPS of the video is {:5.2f}, run time: {:5.2f}".format(1 / (time.time() - pre_time),
                                                                          time.time() - pre_time))
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break


        else:
            break


def frame_skip(frames, fps_time, yolo_time, yolo):
    if frames == 0:
        yolo.skip_flag = 0
    elif fps_time > yolo_time:
        yolo.skip_flag = 0
    else:
        yolo.skip_flag = 1


def acc_skip_test():
    yolo = YOLO()
    videofile = 'test1.avi'
    savefile = './data_bin/test1_f1.csv'

    ## base_line initiation
    acc = accuracy_ex_torch.ACCU()
    acc.load('./data_bin/test1_base.csv')
    ## end base line initiation

    cap = cv2.VideoCapture(videofile)

    assert cap.isOpened(), 'Cannot capture source'

    selection = [320, 416, 608]
    frames = 0
    out_fps = 1 / 24
    fps_time = out_fps
    f1_list = np.array([])
    f1_avg_val = 0
    f1_avg_list = np.array([])
    f1_ind_list = np.array([])
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
                # reso = selection[np.random.randint(3)]
                # yolo.model_switch(reso)

                orig_im = yolo.run(frame, frames)

                # print('yolo data: {}'.format(yolo.data))
                f1, frame_ex = acc.get_acc(yolo.data)
                # print('reso:{}'.format(yolo.model.net_info["height"]))
                cv2.imshow("frame", orig_im)
                yolo_time = time.time() - pre_time
                print('fps:{} ,run tim:{}'.format(1 / yolo_time, yolo_time))
                #print("FPS of the video is {:5.2f}".format(frames / (time.time() - start)))
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
            elif fps_time > yolo_time:
                print('yolo prediction')
                pre_time = time.time()
                print('frame:{}'.format(frames))
                # reso = selection[np.random.randint(3)]
                # yolo.model_switch(reso)

                orig_im = yolo.run(frame, frames)
                yolo_time += time.time() - pre_time
                print('fps:{} ,run tim:{}'.format(1/(time.time()-pre_time), time.time()-pre_time))
                # print('yolo data: {}'.format(yolo.data))
                f1, frame_ex = acc.get_acc(yolo.data)
                # print('reso:{}'.format(yolo.model.net_info["height"]))
                cv2.imshow("frame", orig_im)
                #yolo_time += time.time() - pre_time

                # print("FPS of the video is {:5.2f}".format(frames / (time.time() - start)))
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
            else:
                print('skip')
                fps_time += out_fps
                # print('frame_ex:{}'.format(frame_ex))
                f1, frame_ex = acc.get_acc_skip(frame_ex, frames)

                cv2.imshow("frame", frame)

            frames += 1
            #f1_avg_val, f1_list = f1_avg(f1, f1_list, 29)
            if frames % 100 == 0:

                exists = os.path.isfile(savefile)
                f1_ind_list = np.append(f1_ind_list, f1)
                f1_df = pd.DataFrame({'f1_score': f1_ind_list})
                if exists == True:
                    with open(savefile) as f:

                        read = pd.read_csv(f)
                        read = pd.concat([read, f1_df])
                        read.to_csv(savefile, encoding='utf-8', index=False)
                else:


                    f1_df.to_csv(savefile, encoding='utf-8', index =False)
                f1_ind_list = np.array([])



            f1_ind_list = np.append(f1_ind_list, f1)
            if f1_avg_val != 0:
                f1_avg_list = np.append(f1_avg_list, f1_avg_val)
        else:

            break


def fetch_base_loop():
    folder = './video_data/'
    format = '.mp4'
    for name in glob.glob(folder + '*' + format):
        # print(name)
        label = name.replace(folder, '').replace(format, '')
        fetch_base(name, folder + label + 'base')


if __name__ == '__main__':
    # fetch_base()
    #fetch_base_loop()
    # acc_test()
    acc_skip_test()
    #print('git test')






