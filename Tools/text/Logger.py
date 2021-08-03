import  os
import pandas as pd

#================================#
##### log input in file ##########
#================================#
def log_input_text(data, name, dirName):
    path = os.path.join(dirName, name)
    if not os.path.exists(path):
        os.mkdir(path)
        print("Directory " , path ,  " Created ")
    else:    
        print("Directory " , path ,  " already exists")

    path = os.path.join(path, "data.csv")
    data = pd.DataFrame(data,columns=['text'])
    try:
        os.open(path, os.O_RDONLY)
        data_csv = pd.read_csv(path)
        data_csv = data_csv.append(data,ignore_index=True)
        data_csv.to_csv(path, index=False)
    except: 
        data.to_csv(path, index=False)
