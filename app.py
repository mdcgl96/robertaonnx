from flask import Flask, request
from flask.helpers import send_file
import torch
import numpy as np
from transformers import RobertaTokenizer
import onnxruntime

app = Flask(__name__, static_url_path='/', static_folder='web')

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
ort_session = onnxruntime.InferenceSession("roberta-sequence-classification-9.onnx")

def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )

@app.route("/")
def indexPage():
    return send_file("web/index.html")    

@app.route("/sentiment", methods=["GET"])
def test():

    text = request.args.get('text')
    if (text == ""):
        return "Please use text parameter in GET URL"

    input_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=True)).unsqueeze(0)  # Batch size 1

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_ids)}
    ort_out = ort_session.run(None, ort_inputs)

    pred = np.argmax(ort_out)
    if(pred == 0):
        result = "Prediction: negative"
    elif(pred == 1):
        result = "Prediction: positive"

    return result