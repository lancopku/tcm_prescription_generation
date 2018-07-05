import numpy as np

class Lang:
    def __init__(self, name):
        self.name= name
        self.word2id = {'PADDING':0,'OOV':1,'SOS':2,'EOS':3}
        self.word2count = {'PADDING':0,'OOV':1,'SOS':1,'EOS':1}
        self.id2word = ['PADDING','OOV','SOS','EOS']
        self.n_words = len(self.id2word)
        self.EOS_token = self.word2id['EOS']
        self.SOS_token = self.word2id['SOS']
        self.emb = None
        self.use_pre_emb = False

    def add_word(self, word):
        if word not in self.word2id:
            self.word2id[word] = self.n_words
            self.word2count[word] = 0
            self.id2word.append(word)
            self.n_words += 1
        self.word2count[word] += 1

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)


    def init_emb(self):
        self.emb = [np.zeros(100)] + [np.random.normal(loc=0.,scale=0.1,size=(100)) for i in range(3)]
        self.use_pre_emb = True
