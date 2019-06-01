#from __future__ import division, print_function


import numpy as np
import pandas as pd

import time


class ACCU(object):

    def __init__(self):
        ##################utilities initialization
        self.init_time =time.time()
        self.count = 0
        self.f1List = np.array([])
        self.f1ListLong = np.array([])
        self.data_list = np.array([]).reshape((0, 6))
        self.df = 0


        #print(self.df.head())
        #self.df1 = pd.DataFrame([])
        self.frame_count = 0
        self.avgCount = 0
        self.ten = 0
        self.tenAvgList = np.array([])
        self.tenAvgCount = 0
        self.thirtyAvg = 0
        self.thirtyAvgList = np.array([])
        self.thirtyFrameAvgCount = 0
        self.four_sec_avg = 0
        self.four_sec_list = np.array([])
        self.four_sec_frame_count = 0
        self.state_acc_out = 0

        self.state_avg_list = np.array([])
        self.frameInd = 0
        self.skip = 0
        self.frame = 0
        self.acc_out = False
        self.last_out = 0

    def load(self, base_line):
        self.df = pd.read_csv(base_line)

    def check(self):
        print('check')

    def get_frame(self, frame_ind):
        self.frame = int(frame_ind)

    def intersection_check(self, x0, x1, xp0, xp1):
        # if(xp0 > x0 and xp0 < x1 or xp1 > x0 and xp1 < x1 or xp0 > x0 and xp1< x1):
        if xp0 < x0 and xp1 < x0 or xp0 > x1 and xp1 > x1:
            # return True
            return False
        else:
            # return False
            return True

    def iou_match(self, com, updateFrame):
        # print("original updateFrame:{}".format(updateFrame))
        mask = com.iloc[5] == updateFrame.iloc[:, 5]

        classList = updateFrame[mask]

        # updateFrame.reset_index(inplace=True)
        # updateFrame.drop('frame', axis=1, inplace=True)
        # print("reset updateFrame:{}".format(updateFrame))

        if classList.empty == True:
            match = 0
            return match, updateFrame

        # print(mask)
        # print(classList)
        # print(com)
        # print(updateFrame)

        comx = com.iloc[1:3].array
        comx = comx.astype(float)
        comy = com.iloc[3:5].array
        comy = comy.astype(float)
        size = len(classList)
        # print(size)
        # print(comx)
        # print(comy)
        # print(classList)
        IoUList = np.array([])
        for i in range(size):
            basex = classList.iloc[i, 1:3].array
            basey = classList.iloc[i, 3:5].array
            x_dir = self.intersection_check(basex[0], basex[1], comx[0], comx[1])
            y_dir = self.intersection_check(basey[0], basey[1], comy[0], comy[1])

            if (x_dir == True and y_dir == True):
                diffx = comx[0] - basex[0]
                diffy = comy[0] - basey[0]
                if (diffx > 0):
                    interWidth = basex[1] - comx[0]
                else:
                    interWidth = comx[1] - basex[0]
                if (diffy > 0):
                    interHeight = basey[1] - comy[0]
                else:
                    interHeight = comy[1] - basey[0]
            else:
                interWidth = 0;
                interHeight = 0;
            # if(interWidth !=0 and interHeight != 0):
            interArea = interWidth * interHeight
            baseArea = (basex[1] - basex[0]) * (basey[1] - basey[0])
            comArea = (comx[1] - comx[0]) * (comy[1] - comy[0])
            UnionArea = baseArea + comArea - interArea
            IoU = interArea / UnionArea
            #print(basex[0], basex[1], basey[0], basey[1])
            #print('interArea:{}, baseArea:{}, comArea:{}, UnionArea:{}, IoU:{}'.format(interArea, baseArea, comArea, UnionArea, IoU))
            IoUList = np.append(IoUList, IoU)
            # else:
            # IoU=0
            # IoUList=np.append(IoUList,IoU)
        # print(IoUList)
        indMax = np.argmax(IoUList)
        # print(indMax)
        ioUMax = IoUList[indMax]
        # print(ioUMax)
        # classList.reset_index(inplace=True)
        # print(classList)
        if (ioUMax >= 0.50):
            newFrame = updateFrame.drop(updateFrame.index[indMax])
            # newclassList.reset_index(inplace=True)
            # newclassList.drop(newclassList.columns[0], axis=1, inplace=True)
            # newclassList.set_index("frame", inplace=True)
            match = 1
            # print('updatedFrame:{}'.format(newFrame))
            print('IoUMax:{}'.format(ioUMax))
            return match, newFrame
        else:
            match = 0
            return match, updateFrame

        # print(IoUList[indMax])



    def get_acc(self, frame_data):

        pre_frame =frame_data
        #df1 = pd.DataFrame({'frame':frame_data[:,0], 'x0':frame_data[:,1], 'x1':frame_data[:, 2], 'y0':frame_data[:, 3], 'y1':frame_data[:, 4], 'class':frame_data[:, 5]})
        df1 = pd.DataFrame(
            {'frame': frame_data[:, 0], 'x0': frame_data[:, 1], 'x1': frame_data[:, 2], 'y0': frame_data[:, 3],
             'y1': frame_data[:, 4]})
        df1['class'] = frame_data[:, 5]
        df = self.df
        #print(df)

        if df1.empty == True:
            print('df1 is empty. frame:{}'.format(df))
            f1 = 0
            return f1

        # # position notes:
        # # 0: x0 , 1: x1, 2: y0, 3: y1, 4: class

        frame = df
        frame1 = df1
        matchTotal = 0
        ## getting matched frame from current frame
        mask = int(frame1.iloc[-1, 0]) == frame.iloc[:, 0].astype('int64')
        frame_detect = frame[mask]
        frame_list = frame[mask]
        ## getting last frame list
        mask = int(frame1.iloc[-1, 0]) == frame1.iloc[:, 0].astype('int64')
        frame1 = frame1[mask]
        #print('frame1:{}'.format(frame1))
        #print('frame_list"{}'.format(frame_list))
        #print('frame1 len:{}'.format(len(frame1)))

        for i in range(len(frame1)):
             if frame_list.empty:
                 break
             #print('frame1:{}'.format(frame1))
             #print('remove list:{}'.format(frame_list))
             match, frame_list = self.iou_match(frame1.iloc[i, :], frame_list)
             #print('remove list:{}'.format(frame_list))
             #print('match:{}'.format(match))
             #print('frame list:{}'.format(frame_list))
             matchTotal += match
             #print('matchTotal:{}'.format(matchTotal))

        size_r = len(frame_detect)
        size_p = len(frame1)
        #print('size r:{}, size p:{}'.format(size_r, size_p))
        if size_p != 0:

            pre = matchTotal / size_p
        else:
            pre = 0

        if size_r != 0:
            rec = matchTotal / size_r

        else:
            rec = 0

        #print('match total:{}, pre:{}, rec:{}'.format(matchTotal, pre, rec))
        if pre + rec == 0:
            print('check 1')
            f1 = 0
            return f1, pre_frame
        #
        #     # print('com size:{}, base size:{}, matchTotal:{}'.format(sizep, sizer, matchTotal))
        #     # print('precision:{}, recall:{}, f1 score:{}'.format(pre, rec, f1))


        f1 = 2 * (pre * rec) / (pre + rec)
            #print('com size:{}, base size:{}, matchTotal:{}'.format(sizep, sizer, matchTotal))
        #print('precision:{}, recall:{}, f1 score:{}'.format(pre, rec, f1))

        return f1, pre_frame

    def get_acc_skip(self, frame_data, frame_ind):
        #print('check')
        pre_frame =frame_data
        #df1 = pd.DataFrame({'frame':frame_data[:,0], 'x0':frame_data[:,1], 'x1':frame_data[:, 2], 'y0':frame_data[:, 3], 'y1':frame_data[:, 4], 'class':frame_data[:, 5]})
        df1 = pd.DataFrame(
            {'frame': frame_data[:, 0], 'x0': frame_data[:, 1], 'x1': frame_data[:, 2], 'y0': frame_data[:, 3],
             'y1': frame_data[:, 4]})
        df1['class'] = frame_data[:, 5]
        df = self.df
        #print(df)
        if df1.empty == True:
            #print('df1 is empty. frame:{}'.format(df))
            f1 = 0
            return f1

        # # position notes:
        # # 0: x0 , 1: x1, 2: y0, 3: y1, 4: class

        frame = df
        frame1 = df1
        matchTotal = 0
        ## getting matched frame from current frame
        mask = frame_ind == frame.iloc[:, 0].astype('int64')
        frame_detect = frame[mask]
        frame_list = frame[mask]
        ## getting last frame list
        #print('frame1:{}'.format(frame1))
        #print('frame_list"{}'.format(frame_list))
        #print('frame1 len:{}'.format(len(frame1)))
        mask = int(frame1.iloc[-1, 0]) == frame1.iloc[:, 0].astype('int64')
        frame1 = frame1[mask]
        for i in range(len(frame1)):
             if frame_list.empty:
                 break
             #print('frame1:{}'.format(frame1))
             #print('remove list:{}'.format(frame_list))
             match, frame_list = self.iou_match(frame1.iloc[i, :], frame_list)

             matchTotal += match

        size_r = len(frame_detect)
        size_p = len(frame1)

        #print('size r:{}, size p:{}, matchTotal:{}'.format(size_r, size_p, matchTotal))
        if size_p != 0:
            pre = matchTotal / size_p
        else:
            pre =0

        if size_r != 0:
            rec = matchTotal / size_r
        else:
            rec=0
        #print('match total:{}, pre:{}, rec:{}'.format(matchTotal, pre, rec))

        if pre + rec == 0:
            f1 = 0
            return f1, pre_frame
        #
        #print('com size:{}, base size:{}, matchTotal:{}'.format(sizep, sizer, matchTotal))
        #print('precision:{}, recall:{}, f1 score:{}'.format(pre, rec, f1))


        f1 = 2 * (pre * rec) / (pre + rec)
            #print('com size:{}, base size:{}, matchTotal:{}'.format(sizep, sizer, matchTotal))
        #print('precision:{}, recall:{}, f1 score:{}'.format(pre, rec, f1))

        return f1, pre_frame

    def avg_acc(self, f1List):
        avg = np.average(f1List)

        return avg


    def frame_avg(self, avgList, frameCount, n):
        avg = 0
        if len(avgList) == n:
            # print('avgList before sum:{}'.format(avgList))
            avg = np.average(avgList)
            avgList = np.array([])
            frameCount = 0
            return avgList, avg, frameCount
        else:
            frameCount += 1
            avg = None
            # print('avgList:{}'.format(avgList))
            return avgList, avg, frameCount

    def get_baseline(self, csv_file):
        self.df =csv_file

    def get_action(self, action):
        self.action = action


    def accuracy_update(self, video_frame_cnt):
        #print('df1:{}'.format(self.df1))

        #10 frames avearge
        self.f1List, self.ten, self.frame_count = self.frame_avg(self.f1List, self.frame_count, 10)
        #print('f1list:{}, ten:{}, frame count:{}'.format(self.f1List, self.ten, self.frame_count))
        if self.ten != None:
            self.tenAvgList = np.append(self.tenAvgList, self.ten)


        # 30 frames avearage
        self.tenAvgList, self.thirtyAvg, self.tenAvgCount = self.frame_avg(self.tenAvgList, self.tenAvgCount, 3)
        #print('tenAvgList:{}, thirtyAvg:{}, tenAvgCount:{}'.format(self.tenAvgList, self.thirtyAvg, self.tenAvgCount))
        if self.thirtyAvg != None:
            self.thirtyAvgList = np.append(self.thirtyAvgList, self.thirtyAvg)

        #print('frame:{}, thirtyAVGList:{}'.format(self.frame, self.thirtyAvgList))
        # 120 frames avearage
        self.thirtyAvgList, self.four_sec_avg, self.thirtyFrameAvgCount = self.frame_avg(self.thirtyAvgList, self.thirtyFrameAvgCount, 4)
        #print('thirtyAvgList:{}, four_sec_avg:{}, thirtyFrameAvgCount:{}'.format(self.thirtyAvgList, self.four_sec_avg, self.thirtyFrameAvgCount))
        if self.four_sec_avg != None:
            self.four_sec_list = np.append(self.four_sec_list, self.four_sec_avg)
            ## state accuracy output
            self.acc_out = True
            print(' ready to output, frame:{}, state acc output:{}'.format(self.frame, self.four_sec_avg))
        elif len(self.four_sec_list) < 4:
            self.acc_out = False

        #last frame condition
        if self.frame == video_frame_cnt - 1:
            print("last frame")
        #     self.f1List, self.ten, self.frame_count = self.frame_avg(self.f1List, self.frame_count, len(self.f1List))
        #     if self.ten != None and self.thirtyAvg == None and self.four_sec_avg == None:
        #         self.tenAvgList = np.append(self.tenAvgList, self.ten)
        #         self.acc_out = True
        #         self.four_sec_avg = self.ten
        #         print(' last frame output, frame:{}, state acc output:{}'.format(self.frame, self.last_out))
        #     # print('frame:{}, minAVGList:{}'.format(self.frame, self.minAvgList))
        #     # 30 frames avearage
        #     self.tenAvgList, self.thirtyAvg, self.tenAvgCount = self.frame_avg(self.tenAvgList, self.tenAvgCount, len(self.tenAvgList))
        #     if self.thirtyAvg != None and self.four_sec_avg ==  None:
        #         self.thirtyAvgList = np.append(self.thirtyAvgList, self.thirtyAvg)
        #         self.four_sec_avg = self.thirtyAvg
        #         self.acc_out = True
        #         print(' last frame output, frame:{}, state acc output:{}'.format(self.frame, self.last_out))
        #     # 120 frames avearage
        #     self.thirtyAvgList, self.four_sec_avg, self.thirtyFrameAvgCount = self.frame_avg(self.thirtyAvgList,
        #                                                                                     self.thirtyFrameAvgCount, len(self.thirtyAvgList))
            if self.four_sec_avg != None:
                self.four_sec_list = np.append(self.four_sec_list, self.four_sec_avg)
                self.acc_out = True
                print(' last frame output check 1, frame:{}, state acc output:{}'.format(self.frame, self.four_sec_avg))

            elif self.thirtyAvg != None:
                self.thirtyAvgList = np.append(self.thirtyAvgList, self.thirtyAvg)
                self.four_sec_avg = self.thirtyAvg
                self.acc_out = True
                print(' last frame output check 2, frame:{}, state acc output:{}'.format(self.frame, self.last_out))
            elif self.ten != None:
                self.tenAvgList = np.append(self.tenAvgList, self.ten)
                self.acc_out = True
                self.four_sec_avg = self.ten
                print(' last frame output check 3, frame:{}, state acc output:{}'.format(self.frame, self.last_out))
            else:
                self.four_sec_avg = 0
                self.acc_out = True
                print(' last frame output check 3, frame:{}, state acc output:{}'.format(self.frame, self.last_out))

    def state_accuracy(self):

        self.df1 = pd.DataFrame(
                {'x0': self.data_list[:, 1], 'x1': self.data_list[:, 2], 'y0': self.data_list[:, 3],
                 'y1': self.data_list[:, 4], 'class': self.data_list[:, 5]})

        if self.df1.empty != True:
            self.f1List = np.append(self.f1List, self.get_acc(self.df.loc[self.frame, :], self.df1, self.frame))
            self.f1ListLong = np.append(self.f1List, self.get_acc(self.df.loc[self.frame, :], self.df1, self.frame))

    def state_accuracy_1(self, data_list, frame):

        self.df1 = pd.DataFrame(
                {'x0': data_list[:, 1], 'x1': data_list[:, 2], 'y0': data_list[:, 3],
                 'y1': data_list[:, 4], 'class': data_list[:, 5]})
        #print('df1:{}'.format(self.df1))
        if self.df1.empty != True:
            self.f1List = np.append(self.f1List, self.get_acc_1(self.df.loc[frame, :], self.df1, frame))
            self.f1ListLong = np.append(self.f1List, self.get_acc_1(self.df.loc[frame, :], self.df1, frame))
        else:
            self.f1List = np.append(self.f1List, 0)
            self.f1ListLong = np.append(self.f1List, 0)

    def run(self, yolov_pred, frameind):

        self.get_acc(self, yolov_pred, frameind)


if __name__ == '__main__':
   accuracy = ACCU()
   accuracy.check