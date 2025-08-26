import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Seq2Seq(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers=1):
        super(Seq2Seq, self).__init__()
        self.len_output = output_dim

        # Encoder
        self.input_embedding = nn.Embedding(input_dim, hidden_dim)
        self.encoder = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)

        # Decoder
        self.output_embedding = nn.Embedding(output_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        src: (batch, src_len)
        trg: (batch, trg_len)
        """
        batch_size, trg_len = trg.shape
        outputs = torch.zeros(batch_size, trg_len, self.len_output).to(device)

        # ---- Encoder ----
        src_embedded = self.input_embedding(src)  # (batch, src_len, hidden_dim)
        _, (hidden, cell) = self.encoder(src_embedded)

        # ---- First decoder input is <SOS> ----
        input_token = trg[:, 0].unsqueeze(1)  # (batch, 1)

        # ---- Decoder loop ----
        for t in range(1, trg_len):
            input_embedded = self.output_embedding(input_token)  # (batch, 1, hidden_dim)
            output, (hidden, cell) = self.decoder(input_embedded, (hidden, cell))
            prediction = self.fc_out(output.squeeze(1))  # (batch, output_dim)
            outputs[:, t, :] = prediction

            # Decide if we use teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = prediction.argmax(1).unsqueeze(1)  # (batch, 1)

            input_token = trg[:, t].unsqueeze(1) if teacher_force else top1

        return outputs
