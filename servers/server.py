import torch
import torch.nn as nn
import transformers

from spell.serving import BasePredictor

class TwitterSentimentExtractionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        # configuring the model to output hidden states
        cfg = transformers.PretrainedConfig.get_config_dict("bert-base-uncased")[0]  # tuple?
        cfg["output_hidden_states"] = True
        cfg = transformers.BertConfig.from_dict(cfg)
        
        bert = transformers.BertModel.from_pretrained("bert-base-uncased", config=cfg)
        self.bert = bert
        self.drop_out = nn.Dropout(0.1)
        self.l0 = nn.Linear(768 * 2, 2)
        torch.nn.init.normal_(self.l0.weight, std=0.02)
        # self.out = nn.LogSoftmax(dim=-2)
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        # ignore the output from the model head, we'll instead we'll construct our own attention
        # head connected to the last two layers of hidden weights.
        # that's 512x762x2=780288 connections.
        _, _, out = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        out = torch.cat((out[-1], out[-2]), dim=-1)
        out = self.drop_out(out)
        # the new head uses a linear layer with two output nodes.
        # the first node learns sequence start.
        # the second node learns sequence end.
        logits = self.l0(out)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        return start_logits, end_logits

class Predictor(BasePredictor):
    def __init__(self):
        if torch.cuda.device_count() >= 1:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = TwitterSentimentExtractionModel()
        self.model.load_state_dict(torch.load(f"/model/model.pth", map_location=self.device))

    def predict(self, payload):
        # payload = {'attention_mask': 
        #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
        #      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #      0, 0],
        # 'input_ids':
        #     [101, 8699, 102, 1045, 1036, 1040, 2031, 5838, 1010, 2065, 1045,
        #      2020, 2183, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #      0, 0, 0, 0, 0, 0, 0, 0, 0],
        # 'token_type_ids':
        #     [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
        #      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #      0, 0],
        # 'y_first': 4,
        # 'y_last': 13}
        input_ids = torch.tensor([payload['input_ids']]).to(self.device)
        token_type_ids = torch.tensor([payload['token_type_ids']]).to(self.device)
        attention_mask = torch.tensor([payload['attention_mask']]).to(self.device)
        start_logits, end_logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        start_logits = start_logits.detach().tolist()
        end_logits = end_logits.detach().tolist()
        return {"start_logits": start_logits, "end_logits": end_logits}
