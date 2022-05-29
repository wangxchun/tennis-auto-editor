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

def load_annos(annos_file):
    gts = []
    with open(annos_file, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            for bin in line:
                gts.append(int(bin))
    print(len(gts))
    for i in range(15):
        gts.pop(0)
    print(len(gts))

    return gts



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
video = 'test_recall_acc/test.mp4'
annos_file = 'test_recall_acc/list_.txt'
cap = cv2.VideoCapture(video)
retaining = True

gts = load_annos(annos_file)

clip = []
index = 0
while retaining:
    retaining, frame = cap.read()

    if not retaining and frame is None:
        continue

    tmp_ = center_crop(cv2.resize(frame, (171, 128)))

    tmp = tmp_ - np.array([[[90.0, 98.0, 102.0]]])
    clip.append(tmp)

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

        results.append(label)
        clip.pop(0)

        print(index, label, gts[index])
        index += 1

gts = np.array(gts)
results = np.array(results)

l = min(len(gts), len(results))

gts = gts[:l]
results = results[:l]

file = open('./test_recall_acc/list_results.txt','w');
file.write(str(results));
file.close();

print("[test] Acc: {}".format(np.mean((results+gts==2)+(results+gts==0))))
print('recall: ', np.sum(results+gts==2)/np.sum(gts==1))


