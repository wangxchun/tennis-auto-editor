import os
from tqdm import tqdm
import torch
from torch import nn
from network import C3D_model
from glob import glob
import cv2
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def center_crop(frame):
    frame = frame[8:120, 30:142, :]
    return np.array(frame).astype(np.uint8)


num_classes = 2
dataset = 'ucf101'

load_from = 'run/run_29/models/C3D-ucf101_epoch-199.pth.tar'
data_dirs = 'data/test_videos'

model = C3D_model.C3D(num_classes=num_classes)

checkpoint = torch.load(load_from)
model.load_state_dict(checkpoint['state_dict'])

model.to(device)
model.eval()

running_corrects = 0.0
results = []
for video in tqdm(glob(os.path.join(data_dirs, '*'))):
    #video = 'data/test/v008 play_Trim.mp4'
    cap = cv2.VideoCapture(video)
    retaining = True

    clip = []
    while retaining:
        retaining, frame = cap.read()

        if not retaining and frame is None:
            continue
        tmp_ = center_crop(cv2.resize(frame, (171, 128)))

        tmp = tmp_ - np.array([[[90.0, 98.0, 102.0]]])
        clip.append(tmp)

        # cv2.imshow('', np.array(tmp_+np.array([[[90.0, 98.0, 102.0]]]))/255.0)
        # cv2.waitKey(1)

        if len(clip) == 16:
            inputs = np.array(clip).astype(np.float32)
            inputs = np.expand_dims(inputs, axis=0)
            inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
            inputs = torch.from_numpy(inputs)
            inputs = torch.autograd.Variable(inputs, requires_grad=False).to(device)
            with torch.no_grad():
                outputs = model.forward(inputs)

            probs = torch.nn.Softmax(dim=1)(outputs)
            # print(probs)
            label = torch.max(probs, 1)[1].detach().cpu().numpy()[0]

            if (label==0 and 'no' in video) or (label==1 and 'no' not in video):
                results.append(1)
            else:
                results.append(0)
            clip.pop(0)
            print(video, label)


print("[test] Acc: {}".format(np.mean(results)))


