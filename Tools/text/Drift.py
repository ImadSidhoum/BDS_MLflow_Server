from mlflow.tracking import MlflowClient
import  os
import pandas as pd
from alibi_detect.utils.saving import load_detector
from statistics import mean


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


def drift_predict(name,Gauge_drift,dirName):
    print("Drift prediction start")
    model = get_drift_model(name)
    if model:
        print("Model drift found")
        dir = os.path.join(dirName, name)
        data_path = os.path.join(dir,'data.csv')
        data = pd.read_csv(data_path)['text']
        if data.shape[0] > 0:
            print(f"data: {data.shape}")
            preds = model.predict(data)
            print(f"pred: {preds}")
            Gauge_drift.labels(name).set(mean(preds["data"]["distance"]))


def driftText(name,dirName, Gauge_drift):  
    drift_predict(name,Gauge_drift,dirName)