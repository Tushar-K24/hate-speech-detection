import torch
import time
from keras_preprocessing.sequence import pad_sequences
from transformers import BertTokenizerFast, BertConfig, BertModel
from flask import Flask, request, jsonify
from config import CFG
from kimcnn import Net

app = Flask(__name__)

# configure device (cpu or gpu)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("There are %d GPU(s) available." % torch.cuda.device_count())
    print("We will use the GPU:", torch.cuda.get_device_name(0))
else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")

# configure BERT
configuration = BertConfig()

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased", do_lower_case=True)
model = BertModel.from_pretrained("bert-base-uncased", add_pooling_layer=False)
configuration = model.config
model.eval()
model.to(device)

# configure kimCNN
net = Net(CFG.data.input_shape).to(device)
net.load_state_dict(torch.load(CFG.data.PATH, map_location=device))
net.eval()


@app.route("/sentiment", methods=["GET"])
def sentiment():
    sent = request.form["sentence"]
    input_ids = tokenizer.encode(sent, add_special_tokens=True)
    input_ids = pad_sequences(
        [input_ids],
        maxlen=CFG.data.MAX_LEN,
        dtype="long",
        value=0,
        truncating="post",
        padding="post",
    )[0]
    att_mask = (input_ids > 0).astype(int)
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
    att_mask = torch.tensor(att_mask).unsqueeze(0).to(device)
    outputs = model(
        input_ids, token_type_ids=None, attention_mask=att_mask
    ).last_hidden_state
    out = net(outputs.unsqueeze(0))
    containsHateSpeech = int(out[0].argmax())
    return {
        "sentence": sent,
        "containsHateSpeech": containsHateSpeech,
    }, 200


if __name__ == "__main__":
    app.run(debug=True)
