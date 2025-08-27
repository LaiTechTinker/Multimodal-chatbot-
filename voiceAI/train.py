# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# import json
# from data_utils import Preparedata

# filepath = './myintent.json'
# with open(filepath, 'r', encoding='utf-8') as f:
#     file = json.load(f)

# word_class = Preparedata(file)
# inputs, outputs = word_class.build_dataset()

# tensor_input = torch.tensor(inputs, dtype=torch.long)
# tensor_output = torch.tensor(outputs, dtype=torch.long)

# dataset = TensorDataset(tensor_input, tensor_output)
# dataloader = DataLoader(dataset, batch_size=12, shuffle=True)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# len_input = len(word_class.word_in_index)
# len_output = len(word_class.word_out_index)

# # Create inverse index for decoding predictions
# index_to_word = {idx: word for word, idx in word_class.word_out_index.items()}


# class VoiceModel(nn.Module):
#     def __init__(self, input_vocab, output_vocab, hidden, num_layer, num_embedd):
#         super().__init__()
#         self.len_input = input_vocab
#         self.len_output = output_vocab
#         self.hidden = hidden
#         self.num_layer = num_layer
#         self.embed = num_embedd

#         self.input_embedding = nn.Embedding(self.len_input, self.embed)
#         self.output_embedding = nn.Embedding(self.len_output, self.embed)

#         self.encoder = nn.LSTM(self.embed, self.hidden, self.num_layer, batch_first=True)
#         self.decoder = nn.LSTM(self.embed, self.hidden, self.num_layer, batch_first=True)

#         self.Linear = nn.Linear(self.hidden, self.len_output)

#     def forward(self, input, output, teacher_forcing_ratio=0.5):
#         batch_size, trg_len = output.shape  

#         input_embding = self.input_embedding(input)
#         _, (hidden_memo, cell_memo) = self.encoder(input_embding)

#         # Prediction container
#         prediction = torch.zeros(batch_size, trg_len, self.len_output).to(input.device)

#         # First decoder input = <SOS>
#         input_token = output[:, 0].unsqueeze(1)

#         # Loop through target sequence
#         for t in range(1, trg_len):
#             outembedding = self.output_embedding(input_token)
#             out_out, (hidden_memo, cell_memo) = self.decoder(outembedding, (hidden_memo, cell_memo))
#             linear_out = self.Linear(out_out.squeeze(1))  # (batch, vocab_size)

#             prediction[:, t, :] = linear_out

#             teacher_force = torch.rand(1).item() < teacher_forcing_ratio
#             top1 = linear_out.argmax(1).unsqueeze(1)

#             input_token = output[:, t].unsqueeze(1) if teacher_force else top1

#         return prediction


# # Initialize model and training parameters
# model = VoiceModel(len_input, len_output, 256, 2, 128).to(device)
# criterion = nn.CrossEntropyLoss(ignore_index=word_class.word_out_index["<PAD>"])
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# epochs = 200


# # Function to decode a sequence of indices into words
# def decode_sequence(seq):
#     words = []
#     for idx in seq:
#         word = index_to_word.get(idx.item(), "<UNK>")
#         if word == "<EOS>":
#             break
#         if word not in ["<PAD>", "<SOS>"]:
#             words.append(word)
#     return " ".join(words)


# # Training loop with accuracy + sentence printing
# if __name__=="__main__":
#  for epoch in range(epochs):
#     model.train()
#     total_loss = 0
#     correct, total = 0, 0
    

#     for batch_in, batch_out in dataloader:
#         batch_in, batch_out = batch_in.to(device), batch_out.to(device)

#         optimizer.zero_grad()
#         # outputs = model(batch_in, batch_out)
#         teacher_forcing_ratio = max(0.5 * (1 - epoch / epochs), 0.0)
    
#         outputs = model(batch_in, batch_out, teacher_forcing_ratio=teacher_forcing_ratio)

#         # Loss calculation
#         loss = criterion(
#             outputs[:, 1:, :].reshape(-1, len_output),
#             batch_out[:, 1:].reshape(-1)
#         )
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()

#         # Accuracy calculation (ignoring <PAD>)
#         preds = outputs[:, 1:, :].argmax(2)  # (batch, seq_len-1)
#         mask = batch_out[:, 1:] != word_class.word_out_index["<PAD>"]
#         correct += ((preds == batch_out[:, 1:]) & mask).sum().item()
#         total += mask.sum().item()

#     avg_loss = total_loss / len(dataloader)
#     accuracy = correct / total * 100 if total > 0 else 0

#     # Show prediction for the first sample in dataset
#     model.eval()
#     with torch.no_grad():
#         sample_in, sample_out = dataset[0]
#         sample_in = sample_in.unsqueeze(0).to(device)
#         sample_out = sample_out.unsqueeze(0).to(device)

#         pred = model(sample_in, sample_out, teacher_forcing_ratio=0.0)  # no teacher forcing
#         pred_tokens = pred.argmax(2).squeeze(0)  # best word at each step
#         predicted_sentence = decode_sequence(pred_tokens)

#     print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%")
#     print(f"Sample Prediction: {predicted_sentence}\n")
# torch.save(model.state_dict(),'voice_model.pth')
# torch.save({
#     "word_in_index": word_class.word_in_index,
#     "word_out_index": word_class.word_out_index,
#     "index_out_word": word_class.index_out_word,
#     "index_in_word":word_class.index_in_word
# },'word_dict.pth')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
from data_utils import Preparedata

filepath = './myintent.json'
with open(filepath,'r',encoding='utf-8') as f:
    file=json.load(f)

word_class = Preparedata(file)
inputs, outputs = word_class.build_dataset()

tensor_input = torch.tensor(inputs, dtype=torch.long)
tensor_output = torch.tensor(outputs, dtype=torch.long)

dataset = TensorDataset(tensor_input, tensor_output)
dataloader = DataLoader(dataset, batch_size=12, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

len_input = len(word_class.word_in_index)
len_output = len(word_class.word_out_index)
index_to_word = {idx: word for word, idx in word_class.word_out_index.items()}

class VoiceModel(nn.Module):
    def __init__(self, input_vocab, output_vocab, hidden, num_layer, num_embedd):
        super().__init__()
        self.len_input = input_vocab
        self.len_output = output_vocab
        self.hidden = hidden
        self.num_layer = num_layer
        self.embed = num_embedd

        self.input_embedding = nn.Embedding(self.len_input, self.embed)
        self.output_embedding = nn.Embedding(self.len_output, self.embed)
        self.encoder = nn.LSTM(self.embed, self.hidden, self.num_layer, batch_first=True)
        self.decoder = nn.LSTM(self.embed, self.hidden, self.num_layer, batch_first=True)
        self.Linear = nn.Linear(self.hidden, self.len_output)

    def forward(self, input, output, teacher_forcing_ratio=0.5):
        batch_size, trg_len = output.shape
        input_embding = self.input_embedding(input)
        _, (hidden_memo, cell_memo) = self.encoder(input_embding)
        prediction = torch.zeros(batch_size, trg_len, self.len_output).to(input.device)
        input_token = output[:, 0].unsqueeze(1)

        for t in range(1, trg_len):
            outembedding = self.output_embedding(input_token)
            out_out, (hidden_memo, cell_memo) = self.decoder(outembedding, (hidden_memo, cell_memo))
            linear_out = self.Linear(out_out.squeeze(1))
            prediction[:, t, :] = linear_out
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = linear_out.argmax(1).unsqueeze(1)
            input_token = output[:, t].unsqueeze(1) if teacher_force else top1

        return prediction

def decode_sequence(seq):
    words = []
    for idx in seq:
        word = index_to_word.get(idx.item(), "<UNK>")
        if word == "<EOS>": break
        if word not in ["<PAD>", "<SOS>"]:
            words.append(word)
    return " ".join(words)

if __name__ == "__main__":
    model = VoiceModel(len_input, len_output, 256, 1, 128).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=word_class.word_out_index["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 150

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct, total = 0, 0

        for batch_in, batch_out in dataloader:
            batch_in, batch_out = batch_in.to(device), batch_out.to(device)
            optimizer.zero_grad()
            teacher_forcing_ratio = max(0.5 * (1 - epoch / epochs), 0.0)
            outputs = model(batch_in, batch_out, teacher_forcing_ratio)
            loss = criterion(outputs[:, 1:, :].reshape(-1, len_output), batch_out[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            preds = outputs[:, 1:, :].argmax(2)
            mask = batch_out[:, 1:] != word_class.word_out_index["<PAD>"]
            correct += ((preds == batch_out[:, 1:]) & mask).sum().item()
            total += mask.sum().item()

        avg_loss = total_loss / len(dataloader)
        acc = (correct / total * 100) if total > 0 else 0

        # Sample prediction
        model.eval()
        with torch.no_grad():
            sample_in, sample_out = dataset[0]
            sample_in = sample_in.unsqueeze(0).to(device)
            sample_out = sample_out.unsqueeze(0).to(device)
            pred = model(sample_in, sample_out, teacher_forcing_ratio=0.0)
            pred_tokens = pred.argmax(2).squeeze(0)
            sample_sentence = decode_sequence(pred_tokens)
        print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}, Acc: {acc:.2f}%")
        print(f"Sample: {sample_sentence}\n")

    torch.save(model.state_dict(), 'voice_model.pth')
    torch.save({
        "word_in_index": word_class.word_in_index,
        "word_out_index": word_class.word_out_index,
        "index_out_word": word_class.index_out_word,
        "index_in_word": word_class.index_in_word
    }, 'word_dict.pth')
