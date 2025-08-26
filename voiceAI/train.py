import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim   # fixed import
from data_utils import Preparedata
import json

filepath='./myintent.json'
with open(filepath,'r',encoding='utf-8') as f:
    file=json.load(f)

word_class = Preparedata(file)
inputs, outputs = word_class.build_dataset()

tensor_input = torch.tensor(inputs, dtype=torch.long)
tensor_output = torch.tensor(outputs[:5070], dtype=torch.long)

dataset = TensorDataset(tensor_input, tensor_output)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataloader = DataLoader(dataset, batch_size=12, shuffle=True)

len_input = len(word_class.word_in_index)
len_output = len(word_class.word_out_index)

# print(tensor_input[0:10])
# print(tensor_input.shape)
# print(tensor_output.shape)

class VoiceModel(nn.Module):
    def __init__(self, input_vocab, output_vocab, hidden, num_layer, num_embedd):
        super().__init__()
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        self.len_input = len_input
        self.len_output = len_output
        self.hidden = hidden
        self.num_layer = num_layer
        self.embed = num_embedd

        self.input_embedding = nn.Embedding(self.len_input, self.embed)
        self.output_embedding = nn.Embedding(self.len_output, self.embed)

        self.encoder = nn.LSTM(self.embed, self.hidden, self.num_layer, batch_first=True)
        self.decoder = nn.LSTM(self.embed, self.hidden, self.num_layer, batch_first=True)

        self.Linear = nn.Linear(self.hidden, self.len_output)

    def forward(self, input, output, teacher_forcing_ratio=0.5):
     batch_size, trg_len = output.shape  # trg_len = target sequence length

    # ---- Encoder ----
     input_embding = self.input_embedding(input)
     _, (hidden_memo, cell_memo) = self.encoder(input_embding)

    # ---- Prepare prediction tensor ----
     prediction = torch.zeros(batch_size, trg_len, self.len_output).to(input.device)

    # ---- First decoder input is <SOS> ----
     input_token = output[:, 0].unsqueeze(1)  # (batch, 1)

    # ---- Loop through target length ----
     for t in range(1, trg_len):
        outembedding = self.output_embedding(input_token)
        out_out, (hidden_memo, cell_memo) = self.decoder(outembedding, (hidden_memo, cell_memo))
        linear_out = self.Linear(out_out.squeeze(1))  # (batch, vocab_size)

        # Save prediction
        prediction[:, t, :] = linear_out

        # Decide teacher forcing
        teacher_force = torch.rand(1).item() < teacher_forcing_ratio
        top1 = linear_out.argmax(1).unsqueeze(1)

        input_token = output[:, t].unsqueeze(1) if teacher_force else top1

     return prediction


        
       