import os
from PIL import Image
import numpy as np

def load_dir(path, shape=[400, 400], one_channel=True):
    data = []
    for i in os.listdir(path):
        try:
            img = np.array(Image.open(os.path.join(path, i)).resize(shape))
            
            if len(img.shape) != 3:
                img = np.stack((img,)*3, -1)
            if img.shape[-1] != 3:
                continue
            
            if one_channel:
                img = img[:, :, 0]
                
            data.append(img)
        except:
            1
    return np.array(data)

def load_dirs(path, shape=[400, 400]):
    data = []

    for dir in os.listdir(path):
        if os.path.isdir(os.path.join(path, dir)):
            data.extend(load_dir(os.path.join(path, dir)))

    data = np.array(data)
    
    return data