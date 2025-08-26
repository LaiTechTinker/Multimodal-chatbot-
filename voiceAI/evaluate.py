# from train import VoiceModel, device
# import torch
# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer

# # Download required NLTK data (if not already)
# nltk.download("punkt")
# nltk.download("wordnet")

# checkpoint = torch.load("word_indices.pth")
# word_in_index = checkpoint["word_in_indx"]
# word_out_index = checkpoint["word_out_indx"]
# index_in_word = checkpoint["indx_in"]
# index_out_word = checkpoint["indx_out"]

# len_input = len(word_in_index)
# len_output = len(word_out_index)

# model = VoiceModel(
#     input_vocab=len_input,
#     output_vocab=len_output,
#     hidden=256,
#     num_layer=2,
#     num_embedd=128
# ).to(device)


# model.load_state_dict(torch.load("voice_model.pth", map_location=device))
# model.eval()

# lemmatizer = WordNetLemmatizer()


# def preprocess_input(sentence):
#     # tokenize
#     tokens = word_tokenize(sentence.lower())
#     # lemmatize each token
#     tokens = [lemmatizer.lemmatize(tok) for tok in tokens]
#     # convert to indices, unknown words → <UNK> if exists
#     unk_idx = word_in_index.get("<UNK>")
#     indices = [word_in_index.get(tok, unk_idx) for tok in tokens if tok in word_in_index or unk_idx is not None]
#     return indices


# def evaluate(model, input_seq, max_len=20):
#     with torch.no_grad():
#         input_tensor = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device)  # [1, seq_len]

#         sos_idx = word_out_index["<SOS>"]
#         decoder_input = torch.tensor([[sos_idx]], dtype=torch.long).to(device)

#         input_embedded = model.input_embedding(input_tensor)
#         _, (hidden, cell) = model.encoder(input_embedded)

#         outputs = []
#         for _ in range(max_len):
#             decoder_embedded = model.output_embedding(decoder_input)
#             decoder_output, (hidden, cell) = model.decoder(decoder_embedded, (hidden, cell))
#             prediction = model.Linear(decoder_output.squeeze(1))  # [1, vocab_size]

#             top1 = prediction.argmax(1).item()
#             outputs.append(top1)

#             if top1 == word_out_index["<EOS>"]:
#                 break
#             decoder_input = torch.tensor([[top1]], dtype=torch.long).to(device)

#     decoded_sentence = [index_out_word[idx] for idx in outputs if idx in index_out_word]
#     return decoded_sentence


