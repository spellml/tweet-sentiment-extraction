import os
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.optim import lr_scheduler
import transformers
import tokenizers
from transformers import AdamW, AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter

train = pd.read_csv("/mnt/tweet-sentiment-extraction/train.csv")
def process_data(tweet, selected_text, sentiment, tokenizer, max_len):
    # yes this really happens lol, try idx 314
    if pd.isnull(tweet) or pd.isnull(selected_text) or len(tweet) == 0 or len(selected_text) == 0:
        raise ValueError("text or selected_text is nan.")
    
    # get indicial boundaries of substring.
    target_char_idx_start = tweet.index(selected_text)
    target_char_idx_end = target_char_idx_start + len(selected_text)

    # build the character attention mask (used to build the token attention mask)
    char_target_mask = (
        [0] * target_char_idx_start +
        [1] * (target_char_idx_end - target_char_idx_start) +
        [0] * (len(tweet) - target_char_idx_end)
    )
    
    # tokenize
    # `ids` is the token values, `offsets` are the position tuples for the tokes in the str
    tokens_obj = tokenizer.encode(tweet)
    token_ids, token_offsets = tokens_obj.ids, tokens_obj.offsets
    
    # this is the clever bit. recall that the task is to find the subsequence in the sequence
    # exemplifying the given sentiment. to do this we reformulate the input sequence as a
    # question-answer pair, where the sentiment (as a single word) is the question and the
    # sequence as a whole is the answer.
    # 
    # this allows us to use version of the BERT model pretrained on the well-formed and
    # well-studied question-answering task as a surrogate for this task.
    sentiment_id_map = {
        'positive': 3893,
        'negative': 4997,
        'neutral': 8699
    }
    # 101 is [CLS] and 102 is [SEP]. BERT expects Q/A input to be in the form
    # [CLS] [...] [SEP] [...] [SEP]. Cf.
    # https://huggingface.co/transformers/glossary.html#token-type-ids
    # NOTE: the [-1:1] is the excise the start-of-seq and end-of-seq in the tokens
    input_ids = [101] + [sentiment_id_map[sentiment]] + [102] + token_ids[1:-1] + [102]
    
    # BERT expects Q/A pairs to come with a binary mask splitting the pair types
    # NOTE: the mafs excludes start-of-seq and end-of-seq but includes the new end-of-seq
    token_type_ids = [0, 0, 0] + [1] * (len(token_ids) - 2 + 1)

    # pad to max_len and create a corresponding attention mask
    pad_len = max_len - len(input_ids)
    attention_mask = [1] * len(input_ids) + [0] * pad_len
    input_ids = input_ids + [0] * pad_len
    token_type_ids = token_type_ids + [0] * pad_len
    
    # get the index of the first and last token of the target, this is what the model will try
    # to predict! see the notes on the head layer in forward for more info.
    # we add 3 because the first thee elements of the mask are always [CLS] $SENTIMENT [CLS]
    # and always get an attention vector [1 1 1].
    ufunc = lambda first, _: first >= target_char_idx_start and first < target_char_idx_end
    y_pred_mask = [ufunc(*offset) for offset in token_offsets]
    try:
        y_first = 3 + y_pred_mask.index(True)
        y_last = 3 + len(y_pred_mask) - y_pred_mask[::-1].index(True) - 1
    except ValueError:
        # some of the labels are noisy, and the first character in the label does not actually
        # correspond with the first character of any token (e.g. the label is a part-of-a-word
        # instead of a word). I'm going to venture the opinion here that these records 
        # constitute data noise (because, I mean, they are) and should be removed in
        # pre-processing
        raise ValueError(
            f"Found bad selected_text value '{selected_text}' for tweet '{tweet}'."
            f"Make sure to get rid of these in a pre-processing pass."
        )
    
    # convert to torch tensors
    t = lambda seq: torch.tensor(seq, dtype=torch.long)
    input_ids, token_type_ids, attention_mask, y_first, y_last =\
        t(input_ids), t(token_type_ids), t(attention_mask), t(y_first), t(y_last)
    
    # output
    # Unfortunately the PyTorch dataloader relies on pickle, and TIL namedtuples do not play nice
    # with pickle!
    # Record = namedtuple('record', 'input_ids token_type_ids attention_mask y_first y_last')
    # return Record(input_ids, token_type_ids, attention_mask, y_first, y_last)
    return {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
        "y_first": y_first,
        "y_last": y_last
    }

class TwitterSentimentExtractionDataset:
    def __init__(self, df):
        self.df = df
        self.tokenizer = tokenizers.BertWordPieceTokenizer(
            f"/mnt/bert-base-uncased/vocab.txt", lowercase=True
        )
        self.max_len = 128
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        return process_data(
            self.df.text[item],
            self.df.selected_text[item], 
            self.df.sentiment[item],
            self.tokenizer,
            self.max_len
        )

# preprocessing pass; see comments in the previous code cell on why this is necessary
X_train_preprocessing_pass = TwitterSentimentExtractionDataset(train)

bad_idxs, good_idxs, y_firsts, y_lasts = [], [], [], []
for i in range(len(train)):
# for i in tqdm.tqdm(list(range(len(train)))):
    try:
        x = X_train_preprocessing_pass[i]
        y_firsts.append(x['y_first'])
        y_lasts.append(x['y_last'])
        good_idxs.append(i)
    except ValueError:
        print(f"Found bad record at idx {i}.")
        y_firsts.append(None)
        y_lasts.append(None)
        bad_idxs.append(i)

del X_train_preprocessing_pass
train_orig = train
X_train_df = train.iloc[good_idxs].reset_index(drop=True)

class TwitterSentimentExtractionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        # configuring the model to output hidden states
        cfg = transformers.PretrainedConfig.get_config_dict("bert-base-uncased")[0]  # tuple?
        cfg["output_hidden_states"] = True
        cfg = transformers.BertConfig.from_dict(cfg)
        self.bert = transformers.BertModel.from_pretrained("bert-base-uncased", config=cfg)
        
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
        
        # TODO: embed softmax directly into the model arch
        # y_start, y_end = self.out(start_logits), self.out(end_logits)
        # return y_start, y_end

def loss_fn(start_logits, end_logits, start_positions, end_positions):
    ce = nn.CrossEntropyLoss()
    y_pred_first_loss = ce(start_logits, start_positions)
    y_pred_last_loss = ce(end_logits, end_positions)
    y_pred_loss = (y_pred_first_loss + y_pred_last_loss)
    return y_pred_loss

# create model
device = torch.device("cuda")

model = TwitterSentimentExtractionModel()
model.to(device)

# create optimizer; uses weight decay
model_parameters = list(model.named_parameters())
no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
optimizer_parameters = [
    {'params': [p for n, p in model_parameters if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
    {'params': [p for n, p in model_parameters if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
]
optimizer = AdamW(optimizer_parameters, lr=3e-5)

# batch size and epochs globals; setting batch_size = 1 for testing, 64 is too much for the K80
batch_size = 64
epochs = 5

# dataset and dataloader
dataset = TwitterSentimentExtractionDataset(X_train_df)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=1
)

# create scheduler, this one is a transformers module hookup, thanks huggingface
num_train_steps = int(len(dataloader) * epochs)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=num_train_steps
)

# tensorboard writer
writer = SummaryWriter(f'/spell/tensorboards/model_2')

# mixed-precision gradient scaler
scaler = torch.cuda.amp.GradScaler()

# train func for one epoch of training
def train_fn(dataloader, model, optimizer, device, scheduler, epoch_num):
    model.train()
    
    losses = []
    fn = lambda field: records[field].to(device, dtype=torch.long)
    for idx, records in enumerate(dataloader):
        # move the record to GPU
        input_ids = fn("input_ids")
        token_type_ids = fn("token_type_ids")
        attention_mask = fn("attention_mask")
        y_first = fn("y_first")
        y_last = fn("y_last")
        
        model.zero_grad()
        
        # mixed-precision autocast
        with torch.cuda.amp.autocast():        
            y_pred_start_logits, y_pred_end_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            loss = loss_fn(y_pred_start_logits, y_pred_end_logits, y_first, y_last)
            y_pred_starts = torch.softmax(y_pred_start_logits, dim=1).cpu().detach().numpy()
            y_pred_ends = torch.softmax(y_pred_end_logits, dim=1).cpu().detach().numpy()
        
        # mixed-precision scaling
        scaler.scale(loss).backward()        
        # loss.backward()
        
        scaler.step(optimizer)
        scaler.update()
        # optimizer.step()
        scheduler.step()
        
        lv = loss.item()
        losses.append(lv)
        writer.add_scalar('training loss', lv, epoch_num * len(dataloader) + idx)
        
        if idx % 10 == 0:
            print(f"epoch {epoch_num}, batch {idx} training loss: {losses[-1]}")

    return losses


def main():
    checkpoints_dir = "/spell/checkpoints/"
    if not os.path.exists(checkpoints_dir):
        os.mkdir(checkpoints_dir)
    
    for epoch in range(1, epochs + 1):
        losses = train_fn(dataloader, model, optimizer, device, scheduler, epoch)
        avg_loss = np.mean(losses)
        print(f"epoch {epoch}, average training loss: {avg_loss}")
        torch.save(model.state_dict(), f"/spell/checkpoints/model_{epoch}.pth")

if __name__ == "__main__":
    main()
