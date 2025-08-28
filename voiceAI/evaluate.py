from train import VoiceModel, device
import torch
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer



word_mapping = torch.load('word_dict.pth')
word_in_index = word_mapping['word_in_index']
word_out_index = word_mapping['word_out_index']
index_out_word = word_mapping['index_out_word']
index_in_word = word_mapping['index_in_word']
len_input = len(word_in_index)
len_output = len(word_out_index)

model = VoiceModel(len_input, len_output, 256, 1, 128).to(device)
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
            if top1 == word_out_index["<EOS>"]:
                break
            output_ids.append(top1)
            decoder_input = torch.tensor([[top1]], dtype=torch.long).to(device)

    # Clean up the sentence
    decoded_sentence = [
        index_out_word[idx] for idx in output_ids
        if idx in index_out_word and index_out_word[idx] not in ["<EOS>", "<PAD>", "<SOS>"]
    ]

    # Always return something meaningful
    if not decoded_sentence:
        decoded_sentence = ["Sorry,", "I", "did", "not", "understand."]

    return decoded_sentence


if __name__=="__main__":
    while True:
     user_message = input("Your message: ")
     response = evaluate(user_message)
     print('Virtual assistant:', ' '.join(response))
