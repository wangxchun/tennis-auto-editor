import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(os.path.join(current_dir, ".."))

import torch
import numpy as np
from network import C3D_model
import cv2
import imutils
from imutils.object_detection import non_max_suppression
from imutils import paths
from time import time
#import winsound

print('enter')

torch.backends.cudnn.benchmark = True


def CenterCrop(frame, size):
    h, w = np.shape(frame)[0:2]
    th, tw = size
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))

    frame = frame[y1:y1 + th, x1:x1 + tw, :]
    return np.array(frame).astype(np.uint8)


def center_crop(frame):
    frame = frame[8:120, 30:142, :]
    return np.array(frame).astype(np.uint8)

results = []
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

with open('./dataloaders/ucf_labels.txt', 'r') as f:
    class_names = f.readlines()
    f.close()
# init model
model = C3D_model.C3D(num_classes=2)
print(os.listdir('./run/run_29/models/'))
checkpoint = torch.load('./run/run_29/models/C3D-ucf101_epoch-199.pth.tar')

model.load_state_dict(checkpoint['state_dict'])
model.to(device)
model.eval()

# read video
video = os.path.join(current_dir, r'data', str(sys.argv[1]))
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
    if len(clip) == 16:
        inputs = np.array(clip).astype(np.float32)
        inputs = np.expand_dims(inputs, axis=0)
        inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
        inputs = torch.from_numpy(inputs)
        inputs = torch.autograd.Variable(inputs, requires_grad=False).to(device)
        with torch.no_grad():
            outputs = model.forward(inputs)

        probs = torch.nn.Softmax(dim=1)(outputs)
        label = torch.max(probs, 1)[1].detach().cpu().numpy()[0]

        if label == 0:
            results.append(False)
        else:
            results.append(True)

        cv2.putText(frame, class_names[label].split(' ')[-1].strip(), (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255), 1)
        cv2.putText(frame, "prob: %.4f" % probs[0][label], (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255), 1)
        clip.pop(0)

    #cv2.imshow('result', frame)
    cv2.waitKey(30)

cap.release()
cv2.destroyAllWindows()

print("video:" , video)
np.save(os.path.join('results', video.split('/')[-1]+".npy"), results)
