import numpy as np
from mlflow.tracking import MlflowClient
import  os
import cv2
import glob 

from alibi_detect.utils.saving import load_detector

#================================#
############## DRIFT #############
#================================#

def get_drift_model(name):
    path=None
    client = MlflowClient()
    for rm in client.list_registered_models():
        if dict(rm)["name"] == name:
            l = dict(rm)["latest_versions"]
            for elt in l:
                if elt.current_stage == 'Production':
                    path = elt.source 
    print(path)
    if path:
        new_path = os.path.split(path)
        if (new_path[1] == "model"):
            path = new_path[0]
        else:
            return None

        drift_path = os.path.join(path,"drift")
        if os.path.exists(drift_path):
            drift_model = load_detector(drift_path)
            return drift_model
    return None 


def get_saved_data(count_input_log, name, dirName="./saved_data"):
    dir = os.path.join(dirName, name)
    data_path = os.path.join(dir,'*') 
    files = glob.glob(data_path) 
    count_input_log.labels(name).inc(len(files))
    data = [] 
    for f1 in files: 
        img = cv2.imread(f1) 
        img = cv2.resize(img, (32,32), interpolation=cv2.INTER_AREA)
        data.append(img) 

    data = np.array(data)
    data = data.astype('float32') / 255
    return data

def drift_predict(name,Gauge_drift,dirName, count_input_log):
    print("Drift prediction start")
    model = get_drift_model(name)
    if model:
        print("Model drift found")
        data = get_saved_data(count_input_log,name,dirName)
        if data.shape[0] > 0:
            print(f"data: {data.shape}")
            preds = model.predict(data[:,:,:,::-1])
            print(f"pred: {preds}")
            Gauge_drift.labels(name).set(preds["data"]["distance"])

def drift_predict_input(y,name, Gauge_drift_input):
    print("Drift prediction start: input")
    model = get_drift_model(name)
    if model:
        Y = []
        for i in range(len(y)):
            Y.append(cv2.resize(y[i], (32,32), interpolation=cv2.INTER_AREA))
        Y = np.array(Y)
        preds = model.predict(Y)
        print(f"pred: {preds}")
        Gauge_drift_input.labels(name).set(preds["data"]["distance"])

def driftImage(data,name,dirName, Gauge_drift_input, Gauge_drift, count_input_log):
    drift_predict_input(data,name, Gauge_drift_input)
    drift_predict(name,Gauge_drift,dirName, count_input_log)