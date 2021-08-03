from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pydantic import BaseModel
import mlflow
from typing import Optional


from starlette_exporter import PrometheusMiddleware, handle_metrics
import prometheus_client as prom
import time, os


# import image
from Tools.image.Drift import driftImage
from Tools.image.Logger import log_input_image

#import text
from Tools.text.Logger import log_input_text
from Tools.text.Drift import driftText
from Tools.text.Preproccess import get_artifatcs,preproccess

#docker run -it  -p 5001:5001 --mount type=bind,source=$(pwd),target=/app land95/mlflow-server:0.1

os.chdir("volume")

# Initialisation
app = FastAPI()
mlflow.set_tracking_uri("sqlite:///mlruns.db")

# Variable
dirName = "saved_data"#os.path.join("volume","saved_data")

# creattion du folder qui contiendra les input log
if not os.path.exists(dirName):
    os.mkdir(dirName)
    print("Directory " , dirName ,  " Created ")
else:    
    print("Directory " , dirName ,  " already exists")

# Prometheus tracking log
app.add_middleware(PrometheusMiddleware)
app.add_route("/metrics", handle_metrics)
my_inference_time = prom.Gauge('ATOS_Inference_time', 'This is inference time', ['name'])
my_class_prediction = prom.Counter('ATOS_label_pred', 'None',  ['name', 'label'])
count_class_prediction = prom.Counter('ATOS_number_pred', 'None count', ['name'])
Gauge_drift = prom.Gauge('ATOS_drift', 'None', ['name'])
Gauge_drift_input = prom.Gauge('ATOS_drift_input', 'None', ['name'])
count_input_log = prom.Counter('ATOS_number_input_log', 'None count', ['name'])
score_prediction_log = prom.Histogram('ATOS_pred_score', 'None', ['name','label'])

# Cor
app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://grafana:3000'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# requets
class Item(BaseModel):
    data: list
    dataType: Optional[str] = None


@app.get('/')
async def index():
    return "Server Up"

"""
def count_pred_label(y,name,label_list=None):
    class_y = np.argmax(y, axis=-1)
    print(class_y)
    for i in range(class_y.shape[0]):
        if label_list != None:
                my_class_prediction.labels(name, label_list[class_y[i]]).inc()
        my_class_prediction.labels(name, "class "+ str(class_y[i])).inc()
"""

def count_pred_label(y,name,label_list=None):
    labels = np.argmax(y,axis=-1).reshape(-1)
    sc = np.amax(y,axis=-1).reshape(-1)
    for i in range(labels.shape[0]):
        if label_list != None:
            my_class_prediction.labels(name, label_list[labels[i]]).inc()
            score_prediction_log.labels(name,label_list[labels[i]]).observe(sc[i])
            continue
        my_class_prediction.labels(name, "class "+ str(labels[i])).inc()
        score_prediction_log.labels(name, "class "+ str(labels[i])).observe(sc[i])
        


def LogImage(data, name,dirName, Gauge_drift_input, Gauge_drift, count_input_log):
    log_input_image(data, name, dirName)
    driftImage(data,name,dirName, Gauge_drift_input, Gauge_drift, count_input_log)


def LogText(data,name,dirName,Gauge_drift):
    log_input_text(data,name,dirName)
    driftText(name,dirName, Gauge_drift)


#================================#
############ PREDICT #############
#================================#

@app.post('/predict/{name}')
async def predict(name, item:Item):
    stage = 'Production'
    label_list = None
    dataType = item.dataType
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{name}/{stage}")
    data = np.array(item.data)
    start_timer = time.time()
    if dataType=='text': 
        tokenizer,emb_model,label_list = get_artifatcs(name)
        data =  preproccess(data,tokenizer,emb_model,128)
    y = model.predict(data)

    my_inference_time.labels(name).set(time.time()- start_timer)
    count_class_prediction.labels(name).inc(data.shape[0])
    count_pred_label(y,name,label_list)

    print('passed')
    if dataType=="image":
        LogImage(data ,name, dirName, Gauge_drift_input, Gauge_drift, count_input_log)
    elif dataType=="text":
        LogText(item.data,name,dirName,Gauge_drift)
    else:
        print("data type unknown.")

    return y.tolist()