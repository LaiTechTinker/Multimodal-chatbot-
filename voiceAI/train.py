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

len_input = len(word_class.word_in_index)
len_output = len(word_class.word_out_index)

print(tensor_input[0:10])
print(tensor_input.shape)
print(tensor_output.shape)

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

    def forward(self, input, target, teacher_forcing_ratio=0.5):
        batch_size, target_len = target.shape
        outputs = torch.zeros(batch_size, target_len, self.len_output).to(device)

        # Encode input
        input_embedded = self.input_embedding(input)
        _, (hidden, cell) = self.encoder(input_embedded)

        # First decoder input = <SOS> (here we use target[:,0])
        decoder_input = target[:, 0].unsqueeze(1)

        for t in range(1, target_len):
            decoder_embedded = self.output_embedding(decoder_input)
            decoder_output, (hidden, cell) = self.decoder(decoder_embedded, (hidden, cell))
            prediction = self.Linear(decoder_output.squeeze(1))
            outputs[:, t] = prediction

            # Teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = prediction.argmax(1).unsqueeze(1)
            decoder_input = target[:, t].unsqueeze(1) if teacher_force else top1

        return outputs
