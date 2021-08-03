import numpy as np
import time, os
import cv2

#================================#
##### log input in file ##########
#================================#
def log_input_image(data, name, dirName):
    path = os.path.join(dirName, name)
    if not os.path.exists(path):
        os.mkdir(path)
        print("Directory " , path ,  " Created ")
    else:    
        print("Directory " , path ,  " already exists")
    for i in range(data.shape[0]):
        img_path = f"{time.time()}_{i}_img.jpg"
        cv2.imwrite(os.path.join(path, img_path), (data[i]*255).astype(np.uint8)[:,:,::-1])