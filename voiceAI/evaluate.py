# from train import VoiceModel, device
# import torch
# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer

# # Download required NLTK data (if not already)
# nltk.download("punkt")
# nltk.download("wordnet")

# word_mapping=torch.load('word_dict.pth')
# word_in_index=word_mapping['word_in_index']
# word_out_index=word_mapping['word_out_index']
# index_out_word=word_mapping['index_out_word']
# index_in_word=word_mapping['index_in_word']
# len_input=len(word_in_index)
# len_output=len(word_out_index)




# model = VoiceModel(
#     input_vocab=len_input,
#     output_vocab=len_output,
#     hidden=256,
#     num_layer=2,
#     num_embedd=128
# ).to(device)

# model.load_state_dict(torch.load('voice_model.pth',map_location=device))
# model.eval()
# lemmatizer=WordNetLemmatizer()

# def process_input(input):
#     # tokenize the input
#     tokenize_inp=word_tokenize(input.lower())
#     # lemmatize the input
#     tokenize_inp=[lemmatizer.lemmatize(tok)for tok in tokenize_inp]
#     # convert token to indices, unknow words to unk if it exist
#     unk_idx=word_in_index.get("<UNK>")
#     indices=[word_in_index.get(tok,unk_idx)for tok in tokenize_inp if tok in word_in_index or unk_idx is not None]
#     return indices


# def evaluate(input_seq,max_len=30):
#     with torch.no_grad():
#         input_tensor=torch.tensor(process_input(input_seq),dtype=torch.long).unsqueeze(0).to(device)
#         start_indx=word_out_index['<SOS>']
#         decoder_input=torch.tensor([[start_indx]],dtype=torch.long).to(device)
#         input_embedding=model.input_embedding(input_tensor)
#         _,(hidden,cell)=model.encoder(input_embedding)
#         output=[]
#         for _ in range(max_len):
#             decoder_embedded = model.output_embedding(decoder_input)
#             decoder_output, (hidden, cell) = model.decoder(decoder_embedded, (hidden, cell))
#             prediction = model.Linear(decoder_output.squeeze(1))  # [1, vocab_size]

#             top1 = prediction.argmax(1).item()
#             output.append(top1)

#             if top1 == word_out_index["<EOS>"]:
#                 break
#             decoder_input = torch.tensor([[top1]], dtype=torch.long).to(device)
        
#     decoded_sentence = [index_out_word[idx] for idx in output if idx in index_out_word]
#     return decoded_sentence
# while True:
#         user_message=input("Your message:")
#         response=evaluate(user_message)
#         print('Virtual assistant:', ''.join(response))

from train import VoiceModel, device
import torch
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download("punkt")
nltk.download("wordnet")

word_mapping = torch.load('word_dict.pth')
word_in_index = word_mapping['word_in_index']
word_out_index = word_mapping['word_out_index']
index_out_word = word_mapping['index_out_word']
index_in_word = word_mapping['index_in_word']
len_input = len(word_in_index)
len_output = len(word_out_index)

model = VoiceModel(len_input, len_output, 256, 2, 128).to(device)
model.load_state_dict(torch.load('voice_model.pth', map_location=device))
model.eval()
lemmatizer = WordNetLemmatizer()
MAX_LEN = 20

def process_input(input_text):
    tokens = word_tokenize(input_text.lower())
    tokens = [lemmatizer.lemmatize(tok) for tok in tokens]
    unk_idx = word_in_index.get("<UNK>")
    indices = [word_in_index.get(tok, unk_idx) for tok in tokens]
    if len(indices) < MAX_LEN:
        indices += [word_in_index["<PAD>"]] * (MAX_LEN - len(indices))
    else:
        indices = indices[:MAX_LEN]
    return indices

def evaluate(input_seq, max_len=30):
    with torch.no_grad():
        input_tensor = torch.tensor([process_input(input_seq)], dtype=torch.long).to(device)
        start_idx = word_out_index["<SOS>"]
        decoder_input = torch.tensor([[start_idx]], dtype=torch.long).to(device)

        _, (hidden, cell) = model.encoder(model.input_embedding(input_tensor))
        output_ids = []

        for _ in range(max_len):
            decoder_embedded = model.output_embedding(decoder_input)
            decoder_out, (hidden, cell) = model.decoder(decoder_embedded, (hidden, cell))
            logits = model.Linear(decoder_out.squeeze(1))
            top1 = logits.argmax(1).item()
            output_ids.append(top1)
            if top1 == word_out_index["<EOS>"]:
                break
            decoder_input = torch.tensor([[top1]], dtype=torch.long).to(device)

    decoded_sentence = [index_out_word[idx] for idx in output_ids if idx not in [word_out_index["<PAD>"], word_out_index["<SOS>"]]]
    return decoded_sentence
while True:
    user_message = input("Your message: ")
    response = evaluate(user_message)
    print('Virtual assistant:', ' '.join(response))
