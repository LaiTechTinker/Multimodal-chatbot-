
import nltk
from nltk.tokenize import word_tokenize

class Preparedata:
    def __init__(self, file, max_len=20):
        self.file = file
        self.max_len = max_len
        self.inputVocab = []
        self.outputVocab = []
        self.word_in_index = {"<PAD>":0,"<UNK>":1}
        self.word_out_index = {"<PAD>":0,"<UNK>":1,"<SOS>":2,"<EOS>":3}
        self.index_in_word = {0:"<PAD>",1:"<UNK>"}
        self.index_out_word = {0:"<PAD>",1:"<UNK>",2:"<SOS>",3:"<EOS>"}
        self.input = []
        self.output = []
        self.build_vocab()

    def build_vocab(self):
        for intent in self.file['conversation']:
            self.inputVocab.extend(word_tokenize(intent['prompt'].lower()))
            self.outputVocab.extend(word_tokenize(intent['completion'].lower()))

        self.inputVocab = sorted(set(self.inputVocab))
        self.outputVocab = sorted(set(self.outputVocab))

        for i, word in enumerate(self.inputVocab, start=len(self.word_in_index)):
            self.word_in_index[word] = i
            self.index_in_word[i] = word

        for i, word in enumerate(self.outputVocab, start=len(self.word_out_index)):
            self.word_out_index[word] = i
            self.index_out_word[i] = word

    def encode_inputs(self, sentence):
        tokens = word_tokenize(sentence.lower())
        ids = [self.word_in_index.get(w, self.word_in_index["<UNK>"]) for w in tokens]
        if len(ids) < self.max_len:
            ids += [self.word_in_index["<PAD>"]] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]
        return ids

    def encode_outputs(self, sentence):
        tokens = word_tokenize(sentence.lower())
        ids = [self.word_out_index["<SOS>"]]
        ids += [self.word_out_index.get(w, self.word_out_index["<UNK>"]) for w in tokens]
        ids.append(self.word_out_index["<EOS>"])
        if len(ids) < self.max_len:
            ids += [self.word_out_index["<PAD>"]] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]
        return ids

    def build_dataset(self):
        for intent in self.file['conversation']:
            self.input.append(self.encode_inputs(intent['prompt']))
            self.output.append(self.encode_outputs(intent['completion']))
        return self.input, self.output

