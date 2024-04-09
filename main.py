import numpy as np
import cv2
import os
import time
from pytorch_i3d import InceptionI3d
import torch
from charades_dataset import load_rgb_frames, load_flow_frames, video_to_tensor
import videotransforms
from torchvision import transforms

def save_images_to_video(frames_path, video_name, save_path):
    img=[]
    for frame in os.listdir(os.path.join(frames_path)):
        img.append(cv2.imread(os.path.join(frames_path, frame)))
    height,width,layers=img[1].shape
    video = cv2.VideoWriter(os.path.join(save_path, video_name + ".mp4"),-1,30,(width,height))

    for frame in img:
        video.write(frame)

    cv2.destroyAllWindows()
    video.release()
    # h = int(cap.get(cv2.CAP_PROP_FOURCC))
    # codec = chr(h&0xff) + chr((h>>8)&0xff) + chr((h>>16)&0xff) + chr((h>>24)&0xff)

def main():
    rgb_i3d = InceptionI3d(400, in_channels=3)
    rgb_i3d.replace_logits(8)
    rgb_i3d.load_state_dict(torch.load("./newmodels_info/Training1/rgb_anamoly005000.pt", map_location=torch.device("cpu")))
    rgb_i3d.eval()
    flow_i3d = InceptionI3d(400, in_channels=2)
    flow_i3d.replace_logits(8)
    flow_i3d.load_state_dict(torch.load("./newmodels_info/Training1/flow_anamoly005000.pt", map_location=torch.device("cpu")))
    flow_i3d.eval()
    cropTransform = transforms.Compose([videotransforms.CenterCrop(224)])
    classes = ["Arson", "Assault", "Fighting", "RoadAccidents", "Shooting", "Stealing", "Vandalism", "Normal"]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)

    print("WIDTH:",cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("HEIGHT:",cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("FPS:",cap.get(cv2.CAP_PROP_FPS))
    print("FRAME_COUNT:",cap.get(cv2.CAP_PROP_FRAME_COUNT))

    save_dir = "temp"
    recording = False
    temp_name = ""
    temp_frame_count = 0

    while True:
        ret, frame = cap.read()
        
        resize = cv2.resize(frame, (224, 224))

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        if(recording == True):
            cv2.imwrite(os.path.join(save_dir, temp_name, temp_name + "-%s.jpg" % str(temp_frame_count).zfill(6)), resize)
            temp_frame_count += 1
        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
        if cv2.waitKey(1) == ord(' ') and recording == False:
            recording = True
            temp_name = str(time.time())
            print("Start recording to", temp_name)
            os.makedirs(os.path.join(save_dir, temp_name), exist_ok=True)
        elif cv2.waitKey(1) == ord(' ') and recording == True:
            recording = False
            print("Finished recording to", temp_name)

            rgb_imgs = load_rgb_frames(save_dir, temp_name, 0, temp_frame_count)
            rgb_imgs = cropTransform(rgb_imgs)
            rgb_input = video_to_tensor(rgb_imgs)
            rgb_input = rgb_input[None] # batch size of 1

            with torch.no_grad():
                rgb_prediction = rgb_i3d(rgb_input)
            # prediction = (rgb_prediction + flow_prediction)/2.0
            prediction = rgb_prediction
            predicted_class = torch.max(prediction, dim=2)[0].argmax(1).item()
            print(classes[predicted_class])
            temp_name = ""
            temp_frame_count = 0

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()