import os,sys
import numpy as np
import cv2
from PIL import Image
from multiprocessing import Pool
import argparse
from IPython import embed #to debug
# import skvideo.io
# import scipy.misc


def ToImg(raw_flow,bound):
    '''
    this function scale the input pixels to 0-255 with bi-bound

    :param raw_flow: input raw pixel value (not in 0-255)
    :param bound: upper and lower bound (-bound, bound)
    :return: pixel value scale from 0 to 255
    '''
    flow=raw_flow
    flow[flow>bound]=bound
    flow[flow<-bound]=-bound
    flow-=-bound
    flow*=(255/float(2*bound))
    return flow

def save_flows(flows,image,video_name,save_dir,num,bound):
    '''
    To save the optical flow images and raw images
    :param flows: contains flow_x and flow_y
    :param image: raw image
    :param save_dir: save_dir name (always equal to the video id)
    :param num: the save id, which belongs one of the extracted frames
    :param bound: set the bi-bound to flow images
    :return: return 0
    '''
    #rescale to 0~255 with the bound setting
    flow_x=ToImg(flows[...,0],bound)
    flow_y=ToImg(flows[...,1],bound)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(os.path.join(save_dir, video_name)):
        os.makedirs(os.path.join(save_dir, video_name))

    #save the image
    save_img=os.path.join(save_dir,'img_{:05d}.jpg'.format(num))
    # cv2.imwrite(save_img, image)     # save frame as JPEG file
    # scipy.misc.imsave(save_img,image)

    #save the flows
    # save_x=os.path.join(save_dir,'flow_x_{:05d}.jpg'.format(num))
    # save_y=os.path.join(save_dir,'flow_y_{:05d}.jpg'.format(num))
    save_x = os.path.join(save_dir, video_name, video_name+'-'+str(num).zfill(6)+'x.jpg')
    save_y = os.path.join(save_dir, video_name, video_name+'-'+str(num).zfill(6)+'y.jpg')
    # flow_x_img=Image.fromarray(flow_x)
    flow_x_img=flow_x
    # flow_y_img=Image.fromarray(flow_y)
    flow_y_img=flow_y
    cv2.imwrite(save_x, flow_x_img)     # save frame as JPEG file
    cv2.imwrite(save_y, flow_y_img)     # save frame as JPEG file
    # scipy.misc.imsave(save_x,flow_x_img)
    # scipy.misc.imsave(save_y,flow_y_img)
    return 0

def dense_flow(augs):
    # To extract dense_flow images
    # param augs:the detailed augments:
    #     video_path:
    #     video_name: 
    #     save_dir: the destination path's final direction name.
    #     step: num of frames between each two extracted frames
    #     bound: bi-bound parameter
    # return: no returns
    print(augs)
    video_path,video_name,save_dir,step,bound=augs

    # provide two video-read methods: cv2.VideoCapture() and skvideo.io.vread(), both of which need ffmpeg support

    videocapture=cv2.VideoCapture(video_path)
    if not videocapture.isOpened():
        print('Could not initialize capturing! ', video_name)
        exit()
    # try:
    #     videocapture=skvideo.io.vread(video_path)
    # except:
    #     print('{} read error! '.format(video_path))
    #     return 0
    print(video_path)
    # if extract nothing, exit!
    # if len(videocapture)==0:
    #     print('Could not initialize capturing',video_path)
    #     exit()
    # len_frame=len(videocapture)
    len_frame = int(videocapture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_num=0
    image,prev_image,gray,prev_gray=None,None,None,None
    num0=0
    while True:
        #frame=videocapture.read()
        if num0>=len_frame:
            break
        success, frame= videocapture.read()
        num0+=1
        if frame_num==0:
            image=np.zeros_like(frame)
            gray=np.zeros_like(frame)
            prev_gray=np.zeros_like(frame)
            prev_image=frame
            prev_gray=cv2.cvtColor(prev_image,cv2.COLOR_RGB2GRAY)
            frame_num+=1
            # to pass the out of stepped frames
            step_t=step
            while step_t>1:
                #frame=videocapture.read()
                num0+=1
                step_t-=1
            continue

        image=frame
        gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        frame_0=prev_gray
        frame_1=gray
        ##default choose the tvl1 algorithm
        dtvl1=cv2.optflow.DualTVL1OpticalFlow.create()
        flowDTVL1=dtvl1.calc(frame_0,frame_1,None)
        save_flows(flowDTVL1,image,video_name,save_dir,frame_num,bound) #this is to save flows and img.
        prev_gray=gray
        prev_image=image
        frame_num+=1
        # to pass the out of stepped frames
        step_t=step
        while step_t>1:
            #frame=videocapture.read()
            num0+=1
            step_t-=1