import json
import torch
import nltk
import torch.nn as nn
from nltk.tokenize import word_tokenize
# Make sure resources are downloaded once
# nltk.download('punkt')
# nltk.download('wordnet')
filepath='./myintent.json'

with open(filepath,'r',encoding='utf-8') as f:
    file=json.load(f)
# print(file)
# this class will tokenize the data, lemmatize it, encode it, decode it ,embed it also and prepare it for training
class Preparedata:
    def __init__(self,file):
        self.file=file
        self.tokenizeinput=[]
        self.inputVocab=[]
        self.outputVocab=[]
        self.labelVocab=[]
        self.tokenizeout=[]
        # self.intentlabel=[]
        self.word_out_index={"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
        self.word_in_index={"<PAD>": 0, "<UNK>": 1}
        self.label_index={}
        self.index_input={}
        self.index_output={}
    def tokenize(self):
      lemmatizer = nltk.WordNetLemmatizer()
      for intent in self.file['intents']:
        self.inputVocab.extend(intent['patterns'])
        self.outputVocab.extend(intent['responses'])

      for sentence in self.inputVocab:
        tokens = word_tokenize(sentence.lower())
        for w in tokens:
            self.tokenizeinput.append(lemmatizer.lemmatize(w))

      for sentence in self.outputVocab:
        tokens = word_tokenize(sentence.lower())
        for w in tokens:
            self.tokenizeout.append(lemmatizer.lemmatize(w))

      return self.tokenizeinput, self.tokenizeout

    def buildVocab(self):
        self.inputVocab,self.outputVocab=self.tokenize() # this will call the tokenize function and get the tokenized input and output
        for token in self.inputVocab: #this will build vocab input index from the tokenized input
            if token not in self.word_in_index:
                self.word_in_index[token]=len(self.word_in_index)
                self.index_input[len(self.word_in_index)-1]=token
        for token in self.outputVocab: #this will build vocab input index from the tokenized input
            if token not in self.word_out_index:
                self.word_out_index[token]=len(self.word_out_index)
                self.index_output[len(self.word_out_index)-1]=token
        for intent in self.file['intents']:
            if intent['tag'] not in self.label_index:
                self.label_index[intent['tag']]=len(self.label_index)
        # print(self.label_index)
        # print(self.index_output)
        # print(self.index_input)
        # print(len(self.word_in_index))

    def encode_inputs(self,sentence):
        tokens=word_tokenize(sentence.lower())
        encoded=[self.word_in_index.get(w,self.word_in_index["<UNK>"]) for w in tokens]
        # print(encoded)
        return encoded 
    def encode_outputs(self,sentence):
        tokens=word_tokenize(sentence.lower())
        ids=[self.word_out_index["<SOS>"]]
        ids+=[self.word_out_index.get(w,self.word_out_index["<UNK>"]) for w in tokens]
        ids.append(self.word_out_index["<EOS>"])
        # print(ids)
        return ids
    def padding_sequence_(self,sequence):
         self.max_length=max([len(seq) for seq in sequence])
         return [seq +[0]*(self.max_length-len(seq)) for seq in sequence]
    def build_dataset(self):
        self.input=[]
        self.output=[]
        self.buildVocab()
        for intent in self.file['intents']:
            for pattern in intent['patterns']:
                self.input.append(self.encode_inputs(pattern))
            for response in intent['responses']:
                self.output.append(self.encode_outputs(response))

        padded_inputs=self.padding_sequence_(self.input)
        padded_outputs=self.padding_sequence_(self.output)
    
        input_tensor=torch.tensor(padded_inputs,dtype=torch.long)
        output_tensor=torch.tensor(padded_outputs,dtype=torch.long)
        pairs=list(zip(input_tensor,output_tensor))
        print(len(pairs))
        # pairs=torch.stack(padded_inputs,padded_outputs,dim=1,dtype=torch.long)
        # print(input_tensor.shape)
        # print(output_tensor.shape)
        # print(torch.Tensor(pairs).shape)
        # return pairs
       

         

       
wordClass=Preparedata(file)
# wordClass.tokenize()
# wordClass.buildVocab()
wordClass.build_dataset()
