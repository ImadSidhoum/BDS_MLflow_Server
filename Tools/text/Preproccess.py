
import os, pickle
from mlflow.tracking import MlflowClient
from transformers import AutoTokenizer ,AutoModel



def preproccess(text,tokenizer,model,max_len):
    tokenized = tokenizer(text.tolist(), truncation=True, padding='max_length',max_length=max_len, return_tensors='pt')
    input_id = tokenized['input_ids']
    mask = tokenized['attention_mask']
    out = model(input_ids = input_id, attention_mask=mask)['last_hidden_state']
    return  out.detach().numpy()
"""
def predict(text,model,tokenizer,model_emb,max_len,label_list):
    inp =  preproccess(text,tokenizer,model_emb,max_len) 

    y = model.predict(inp[0])
    labels = tf.argmax(y,axis=-1).numpy().squeeze().tolist()
    sc = tf.reduce_max(y,axis=-1).numpy().squeeze().tolist()

    
    words = tokenizer.convert_ids_to_tokens(inp[1])
    response=[]
    scores = []
    for i in range (len(words)):
        if inp[2][i] == 0 :
            break
        response.append({words[i]:label_list[labels[i]]})
        scores.append(sc[i])
    return response,scores
"""
def get_artifatcs(name):
    path=None
    client = MlflowClient()
    for rm in client.list_registered_models():
        if dict(rm)["name"] == name:
            l = dict(rm)["latest_versions"]
            for elt in l:
                if elt.current_stage == 'Production':
                    path = elt.source 
    if path:
        new_path = os.path.split(path)
        if (new_path[1] == "model"):
            path = new_path[0]
        else:
            return None

        drift_path = os.path.join(path,"drift")
        tokenizer_path = os.path.join(path,"tokenizer")
        emb_path = os.path.join(path,"emb")
        label_path = os.path.join(path,"label_list/")
        if os.path.exists(tokenizer_path):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if os.path.exists(emb_path):
            emb_model = AutoModel.from_pretrained(emb_path)
        if os.path.exists(label_path):
            with open (label_path+'label_list', 'rb') as fp:
                label_list = pickle.load(fp)
            
        return tokenizer,emb_model,label_list
"""
def predict_ner(name,text):
    
    stage = 'Production'
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{name}/{stage}")
    tokenizer,emb_model,label_list = get_artifatcs(name)
    print("c'est bon")
    response,scores=predict(text,model,tokenizer,emb_model,128,label_list)
    print(scores)
    return response"""