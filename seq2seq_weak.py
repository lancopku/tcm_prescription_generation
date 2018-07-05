import torch 
import codecs, os, sys
import time, random
from language import *
import pickle
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
import argparse
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import os

use_cuda = torch.cuda.is_available()
MAX_LENGTH = 20
logsoftmax = torch.nn.LogSoftmax()

def eval_bleu(reference, candidate, log_path):
    ref_file = log_path+'/reference.txt'
    cand_file = log_path+'/candidate.txt'
    with codecs.open(ref_file, 'w', 'utf-8') as f:
        for s in reference:
            f.write(" ".join(s)+'\n')
    with codecs.open(cand_file, 'w', 'utf-8') as f:
        for s in candidate:
            f.write(" ".join(s).strip()+'\n')

    temp = log_path + "/result.txt"
    command = "perl multi-bleu.perl " + ref_file + "<" + cand_file + "> " + temp
    os.system(command)
    with open(temp) as ft:
        result = ft.read()
    os.remove(temp)
    return result

def eval_F(reference, candidate, log_path):
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    ref_file = log_path+'/reference.txt'
    cand_file = log_path+'/candidate.txt'
    with codecs.open(ref_file, 'w', 'utf-8') as f:
        for s in reference:
            f.write(" ".join(s)+'\n')
    with codecs.open(cand_file, 'w', 'utf-8') as f:
        for s in candidate:
            f.write(" ".join(s).strip()+'\n')

    total_right = 0.
    total_ref = 0.
    total_can = 0.
    for r,c in zip(reference, candidate):
        r_set = set(r)-{'EOS'}
        c_set = set(c)-{'EOS'}
        right = set.intersection(r_set, c_set)
        total_right += len(right)
        total_ref += len(r_set)
        total_can += len(c_set)
    total_can = total_can if total_can != 0 else 1
    precision = total_right/float(total_can)
    recall = total_right/float(total_ref)
    if precision == 0 or recall == 0:
        F = 0.
    else:
        F = precision*recall*2./(precision+recall)
    return precision, recall, F

def variable_from_sentence(lang, sentence, from_text=False):
    '''
        convert a sentence text into a pytorch Variable
    '''
    if from_text:
        result = []
        for c in sentence:
            if c in lang.word2id:
                result.append(lang.word2id[c])
            else:
                result.append(lang.word2id['OOV'])
        sentence = result
        sentence.append(lang.EOS_token)
    result = Variable(torch.LongTensor(sentence))
    if use_cuda:
        return result.cuda()
    return result

class Encoder(nn.Module):
    def __init__(self, voc_size, emb_size, hidden_size, n_layers=1, bidirectional=False):
        super(Encoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(voc_size, emb_size, padding_idx=0)
        self.gru = nn.GRU(emb_size,  hidden_size, num_layers=self.n_layers, batch_first=False, bidirectional=self.bidirectional)

    def load_emb(self, emb):
        emb = torch.from_numpy(np.array(emb, dtype=np.float32))
        self.embedding.weight.data.copy_(emb)

    def forward(self, input):
        '''
            input :: sl, bs

            return
                output :: sl, bs, nh*directions
                hidden :: n_layers*directions,bs, nh
        '''
        batch_size = input.size()[0]
        init_state = self.initHidden(batch_size)
        output, state = self.encode(input, init_state)
        return output, state
    
    def encode(self, input, hidden):
        '''
            input :: bs, sl

            return
                output :: bs, sl, nh*directions
                hidden :: n_layers*directions,bs, nh
        '''
        mask = torch.gt(input.data,0)
        input_length = torch.sum((mask.long()),dim=1)       # batch first = True, (batch, sl)
        lengths, indices = torch.sort(input_length, dim=0, descending=True)
        _, ind = torch.sort(indices, dim=0)
        input_length = torch.unbind(lengths, dim=0)
        embedded = self.embedding(torch.index_select(input,dim=0,index=Variable(indices)))
        output, hidden = self.gru(pack(embedded, input_length, batch_first=True), hidden)
        output = torch.index_select(unpack(output, batch_first=True)[0], dim=0,index=Variable(ind))*Variable(torch.unsqueeze(mask.float(),-1))
        hidden = torch.index_select(hidden[-1], dim=0, index=Variable(ind))
        #hidden = torch.unbind(hidden, dim=0)
        #hidden = torch.cat(hidden, 1)
        direction = 2 if self.bidirectional else 1
        assert hidden.size() == (input.size()[0],self.hidden_size) and output.size() == (input.size()[0], input.size()[1],self.hidden_size*direction)
        return output, hidden

    def initHidden(self, batch_size):
        bid = 2 if self.bidirectional else 1
        result = Variable(torch.zeros(self.n_layers*bid, batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda()
        return result


class Luong_Attn(nn.Module):
    def __init__(self, input_size, hidden_size, direction):
        super(Luong_Attn, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size*direction)
        self.combine = nn.Linear(input_size+hidden_size*direction, hidden_size)

    def forward(self, x, memory):
        '''
            x :: bs, nh
            memory :: bs, sl, nh
        '''
        h1 = self.linear(x)   # bs, nh -> bs, nh
        bil = torch.sum(torch.unsqueeze(h1,1)*memory,-1)    # bs, sl
        score = F.softmax(bil)      # bs, sl
        c = torch.sum(torch.unsqueeze(score, -1)*memory,1)      # bs, nh
        output = F.tanh(self.combine(torch.cat([x,c],1)))       # bs, nh
        return output, score

class Bah_Attn(nn.Module):
    def __init__(self, input_size, hidden_size, direction):
        super(Bah_Attn, self).__init__()
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size*direction, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

    def forward(self, x, memory):
        '''
            x :: bs, nh
            memory :: bs, sl, nh
            v*tanh(W*([x:memory]))
        '''
        h_x = self.lin1(x)      # bs, nh
        h_m = self.lin2(memory)     # bs, sl, nh
        score = self.v(F.tanh(torch.unsqueeze(h_x,1)+h_m))      # bs, sl, nh -> bs, sl, 1
        score = F.softmax(torch.squeeze(score,-1))     # bs, sl
        context = torch.sum(torch.unsqueeze(score,-1)*memory,1)     # bs, nh
        return context, score    # bs, sl


class Cover(nn.Module):
    def __init__(self, input_size, hidden_size, direction):
        super(Cover, self).__init__()
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size*direction, hidden_size)
        self.lin3 = nn.Linear(1, hidden_size)
        self.v = nn.Linear(hidden_size, 1)
        
    def forward(self, x, memory, cover):
        '''
            x :: bs, nh
            memory :: bs, sl, nh
            cover :: bs, sl
            v*tanh(W*([x:memory:cover*u]))
        '''
        h_x = self.lin1(x)      # bs, nh
        h_m = self.lin2(memory)     # bs, sl, nh
        h_c = self.lin3(torch.unsqueeze(cover, -1))     # bs, sl, nh
        score = self.v(F.tanh(torch.unsqueeze(h_x,1)+h_m+h_c))      # bs, sl, nh -> bs, sl, 1
        score = F.softmax(torch.squeeze(score,-1))     # bs, sl
        context = torch.sum(torch.unsqueeze(score,-1)*memory,1)     # bs, nh
        return context, score    # bs, sl

class Decoder(nn.Module):
    def __init__(self, args, voc_size, emb_size, hidden_size, bidirectional=False):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.voc_size = voc_size
        self.args = args
        self.bidirectional = bidirectional
        self.direction = 2 if self.bidirectional else 1

        self.embedding = nn.Embedding(self.voc_size, self.emb_size, padding_idx=0)
        if args.cover == 'None':
            if args.feed_word_emb == 'None':
                self.gru_cell = nn.GRUCell(self.hidden_size*self.direction, self.hidden_size)
            else:
                self.gru_cell = nn.GRUCell(self.hidden_size*self.direction+self.emb_size, self.hidden_size)
        else:
            if args.feed_word_emb == 'None':
                self.gru_cell = nn.GRUCell(self.hidden_size*(self.direction+1), self.hidden_size)
            else:
                self.gru_cell = nn.GRUCell(self.hidden_size*(self.direction+1)+self.emb_size, self.hidden_size)
        #self.attn = Luong_Attn(hidden_size, hidden_size)
        self.attn = Bah_Attn(hidden_size, hidden_size, self.direction)
        self.attn_cover = Cover(hidden_size, hidden_size, self.direction)
        self.out = nn.Linear(self.hidden_size, self.voc_size)
        if args.cover == 'big':
            self.cover = nn.Linear(self.voc_size, self.hidden_size)
        elif args.cover == 'small':
            self.cover = nn.Linear(1, self.hidden_size)
        self.cover_out = nn.Linear(self.hidden_size, 1)


    def forward(self, word, hidden, encoder_outputs, output_distribution, encoder_attention):
        assert hidden.size() == (word.size()[0], self.hidden_size), (hidden.size(),word.size(),self.hidden_size)
        if self.args.feed_word_emb == 'word':
            word_vec = self.embedding(word)     #  bs, de
        elif self.args.feed_word_emb == 'mean':
            word_vec = torch.mean(self.embedding(output_distribution.long()),1)     #  bs, de
        if self.args.attention_cover:
            context, score = self.attn_cover(hidden, encoder_outputs, encoder_attention)     # (bs, nh) (bs, sl, nh) -> (bs, nh)
        else:
            context, score = self.attn(hidden, encoder_outputs)     # (bs, nh) (bs, sl, nh) -> (bs, nh)
        if self.args.cover != 'None':
            if self.args.cover == 'big':
                cover_hidden = F.tanh(self.cover(output_distribution))
            if self.args.feed_word_emb == 'None':
                hidden = self.gru_cell(torch.cat([context,cover_hidden],1), hidden)
            else:
                hidden = self.gru_cell(torch.cat([context,cover_hidden,word_vec],1), hidden)
        else:
            if self.args.feed_word_emb == 'None':
                hidden = self.gru_cell(context, hidden)
            else:
                hidden = self.gru_cell(torch.cat([context,word_vec],1),hidden)
        pred_word = self.out(hidden)#+torch.squeeze(self.cover_out(cover_hidden),-1)
        return pred_word, hidden, score


    def load_emb(self, emb):
        emb = torch.from_numpy(np.array(emb, dtype=np.float32))
        self.embedding.weight.data.copy_(emb)

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(batch_size,self.hidden_size))
        if use_cuda:
            return result.cuda()
        return result


def one_hot(ids, nclass):
    result = Variable(torch.zeros((ids.size()[0],nclass)))
    if use_cuda: result = result.cuda()
    result.scatter_(1,torch.unsqueeze(ids,-1),1.)
    return result

def multi_hot(ids, nclass):
    '''
        ids :: bs, sl
    '''
    result = Variable(torch.zeros((ids.size()[0],nclass)))    # bs, nv
    padding_index = Variable(torch.zeros((ids.size()[0],1)).long())      # bs, 1
    if use_cuda:
        result = result.cuda()
        padding_index = padding_index.cuda()
    result.scatter_(1,ids,1.)
    result.scatter_(1,padding_index,0.)
    assert result.data[0][0] == 0.
    return result

def cross_entropy(prob, targets, weight):
    H = -logsoftmax(prob) * targets
    return torch.sum(H * weight)

def train(args, inputs, targets, encoder, decoder, optimizer, tgt_lang):
    optimizer.zero_grad()
    input_length = inputs.size()[1]
    target_length = targets.size()[1]
    batch_size = inputs.size()[0]
    assert batch_size == targets.size()[0]
    voc_size = len(tgt_lang.word2id)
    loss = 0.

    # encoding part
    encoder_output, encoder_state = encoder(inputs)

    # decoding part
    decoder_input = Variable(torch.LongTensor([tgt_lang.SOS_token]*batch_size))
    if use_cuda: decoder_input = decoder_input.cuda()
    decoder_hidden = torch.squeeze(encoder_state,0) if encoder_state.dim()==3 else encoder_state
    use_teacher_forcing = True if random.random() < args.teacher_forcing_ratio else False

    decoder_outputs = torch.zeros((batch_size, voc_size))
    attention_outputs = torch.zeros((batch_size, input_length))
    out_weight = Variable(torch.ones(len(tgt_lang.word2id)))
    out_weight[0] = 0.
    out_weight = torch.unsqueeze(out_weight, 0)
    if use_cuda:
        decoder_outputs = decoder_outputs.cuda()
        attention_outputs = attention_outputs.cuda()
        out_weight = out_weight.cuda()
    all_med = F.softmax(multi_hot(targets, len(tgt_lang.word2id)))
    if use_teacher_forcing:
        for time in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output, Variable(decoder_outputs), Variable(attention_outputs))
            if args.mask: decoder_output = decoder_output*(1.-Variable(decoder_outputs))
            if args.soft_loss:
                loss += cross_entropy(decoder_output, (multi_hot(torch.unsqueeze(targets[:,time],1),len(tgt_lang.word2id))+all_med)/2., weight=out_weight)
            else:
                loss += F.cross_entropy(decoder_output, targets[:,time], ignore_index=0)
            decoder_input = targets[:,time]
            attention_outputs += decoder_attention.data
            decoder_outputs.scatter_(1,torch.unsqueeze(targets[:,time],1).data,1.)
    else:
        for time in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output, Variable(decoder_outputs), Variable(attention_outputs))
            if args.mask: decoder_output = decoder_output*(1.-Variable(decoder_outputs))
            attention_outputs += decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if args.soft_loss:
                loss += cross_entropy(decoder_output, (multi_hot(torch.unsqueeze(targets[:,time],1),len(tgt_lang.word2id))+all_med)/2., weight=out_weight)
            else:
                loss += F.cross_entropy(decoder_output, targets[:,time], ignore_index=0)
            decoder_outputs.scatter_(1,topi,1.)
            decoder_input = Variable(torch.squeeze(topi))
            decoder_input = decoder_input.cuda()
    loss.backward()
    optimizer.step()

    return loss.data/target_length

def evaluate(args, encoder, decoder, sentence, src_lang, tgt_lang, from_text=False, max_length=MAX_LENGTH):
    input_variable = variable_from_sentence(src_lang,sentence, from_text)
    input_variable = torch.unsqueeze(input_variable,0)
    input_length = input_variable.size()[1]
    voc_size = len(tgt_lang.word2id)
    encoder_output, encoder_state = encoder(input_variable)

    decoder_input = Variable(torch.LongTensor([tgt_lang.SOS_token]))
    decoder_input = decoder_input.cuda()
    decoder_hidden = torch.squeeze(encoder_state,0) if encoder_state.dim()==3 else encoder_state

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, input_length)

    decoder_outputs = torch.zeros((1, voc_size))
    attention_outputs = torch.zeros((1, input_length))
    if use_cuda: 
        decoder_outputs = decoder_outputs.cuda()
        attention_outputs = attention_outputs.cuda()
    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_output, Variable(decoder_outputs), Variable(attention_outputs))
        if args.mask: decoder_output = decoder_output*(1.-Variable(decoder_outputs))
        attention_outputs += decoder_attention.data
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        decoder_outputs.scatter_(1,topi,1.)
        ni = topi[0][0]
        if ni == tgt_lang.EOS_token:
            decoded_words.append('EOS')
            break
        else:
            if ni != tgt_lang.word2id['OOV']:
                decoded_words.append(tgt_lang.id2word[ni])
        decoder_input = Variable(torch.LongTensor([ni]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    return decoded_words, decoder_attentions[:di+1]

def padding(sequence, length):
    sequence = sequence[:length]
    while len(sequence) < length:
        sequence.append(0)
    return sequence
         
def make_batches(data, batch_size, tgt_lang):
    batch_num = len(data)//batch_size if len(data)%batch_size==0 else len(data)//batch_size+1
    batches = []
    for batch in range(batch_num):
        mini_batch = data[batch*batch_size:(batch+1)*batch_size] 
        en_max_len = max([len(p[0]) for p in mini_batch])
        de_max_len = max([len(p[1]) for p in mini_batch])
        en_mini_batch = [padding(p[0],en_max_len) for p in mini_batch]
        de_mini_batch = [padding(p[1],de_max_len) for p in mini_batch]
        batches.append((en_mini_batch, de_mini_batch))
    return batches

def evaluate_all(args, data, encoder, decoder, src_lang, tgt_lang, from_text):
    hypothsis = []
    reference = []
    for s,t in tqdm(data,disable=not args.verbose):
        result,att = evaluate(args, encoder, decoder, list(s), src_lang=src_lang,tgt_lang=tgt_lang, from_text=from_text)
        hypothsis.append(result)
        if not from_text:
            reference.append([tgt_lang.id2word[word] for word in t])
        else:
            reference.append(t)
    if not from_text:
        precision, recall, F = eval_F(reference, hypothsis, log_path='./log/'+args.log_dir)
    else:
        precision, recall, F = eval_F(reference, hypothsis, log_path='./log/text_'+args.log_dir)
    return precision, recall, F

def train_iters(args,encoder, decoder, train_data, dev_data, test_data, test_data_text, n_iters, src_lang, tgt_lang):
    start_time = time.time()
    parameters = [p for p in encoder.parameters()] + [p for p in decoder.parameters()]
    optimizer = optim.Adam(parameters)

    best_epoch = 0
    best_F = -1.
    for iter in range(1, n_iters+1):
        total_loss = 0.
        if args.debug:
            train_data = train_data[:2]
        for batch in tqdm(train_data, disable=not args.verbose):
            src, tgt = batch
            src = np.array([np.array(s, dtype=np.long) for s in src], dtype=np.long)
            tgt = np.array([np.array(s, dtype=np.long) for s in tgt], dtype=np.long)
            input_variable = Variable(torch.from_numpy(src))
            target_variable = Variable(torch.from_numpy(tgt))
            if use_cuda:
                input_variable = input_variable.cuda()
                target_variable = target_variable.cuda()
            loss = train(args, input_variable, target_variable, encoder, decoder, optimizer, tgt_lang)
            total_loss += loss
        print('epoch %d, total loss %.2f'%(iter,total_loss.cpu()[0]))
        precision, recall ,F = evaluate_all(args, dev_data, encoder, decoder, src_lang=src_lang, tgt_lang=tgt_lang, from_text=False)
        print('Precision %.3f, recall %.3f, F %.3f'%(precision*100, recall*100, F*100))
        sys.stdout.flush()
        
        if best_F < F:
            best_F = F
            best_epoch = iter
            torch.save(encoder.state_dict(),'models/'+args.log_dir+'_encoder.pt',pickle_protocol=3)
            torch.save(decoder.state_dict(),'models/'+args.log_dir+'_decoder.pt',pickle_protocol=3)
    encoder.load_state_dict(torch.load('models/'+args.log_dir+'_encoder.pt'))
    decoder.load_state_dict(torch.load('models/'+args.log_dir+'_decoder.pt'))

    precision, recall ,F = evaluate_all(args, dev_data, encoder, decoder, src_lang=src_lang, tgt_lang=tgt_lang, from_text=False)
    print('Precision %.3f, recall %.3f, F %.3f'%(precision*100, recall*100, F*100))
    test_precision, test_recall ,test_F = evaluate_all(args, test_data, encoder, decoder, src_lang=src_lang, tgt_lang=tgt_lang, from_text=False)
    text_precision, text_recall ,text_F = evaluate_all(args, test_data_text, encoder, decoder, src_lang=src_lang, tgt_lang=tgt_lang, from_text=True)
    print('test precision %.3f, recall %.3f, F value %.3f'%(test_precision*100, test_recall*100, test_F*100))
    print('text precision %.3f, recall %.3f, F value %.3f'%(text_precision*100, text_recall*100, text_F*100))
    sys.stdout.flush()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_size',default=100,type=int)
    parser.add_argument('--hidden_size',default=300,type=int)
    parser.add_argument('--batch_size',default=20, type=int)
    parser.add_argument('--teacher_forcing_ratio',default=0.8,type=float)
    parser.add_argument('--evaluate_method',default='F',type=str)
    parser.add_argument('--gpu',default=0,type=int)
    parser.add_argument('--bidirectional',default=True,action='store_false')
    parser.add_argument('-e','--epoch_num',default=10,type=int)
    parser.add_argument('-l','--log_dir',default='seq2seq',type=str)
    parser.add_argument('-v','--verbose',default=False,action='store_true')
    parser.add_argument('-d','--debug',default=False, action='store_true')
    ####################################################################################################################
    parser.add_argument('-f','--feed_word_emb',default='None',type=str,choices=['word','mean','None'])
    parser.add_argument('-c','--cover',default='None',type=str,choices=['big','None'])
    parser.add_argument('--soft_loss', default=False, action='store_true')
    parser.add_argument('-m','--mask',default=False, action='store_true')
    parser.add_argument('-a','--attention_cover',default=False, action='store_true')
    parser.add_argument('--threshold',default=0.5, type=float)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if use_cuda: torch.cuda.set_device(args.gpu)
    data, test_data_text, med_lang, cure_lang = pickle.load(open('../dataset/prescription_pairs_data.pkl','rb'))
    encoder = Encoder(voc_size=len(cure_lang.word2id), emb_size=args.emb_size, hidden_size=args.hidden_size, bidirectional=args.bidirectional)
    decoder = Decoder(args=args, voc_size=len(med_lang.word2id), hidden_size=args.hidden_size, emb_size=args.emb_size, bidirectional=args.bidirectional)
    encoder.load_emb(cure_lang.emb)
    decoder.load_emb(med_lang.emb)
    if use_cuda:
        encoder.cuda()
        decoder.cuda()
    random.seed(255)
    random.shuffle(data)
    train_data = data[:int(len(data)*0.9)]
    dev_data = data[int(len(data)*0.9):int(len(data)*0.95)]
    test_data = data[int(len(data)*0.95):]
    batches = make_batches(train_data, args.batch_size,med_lang)
    train_iters(args,encoder, decoder, batches, dev_data, test_data, test_data_text, args.epoch_num, tgt_lang=med_lang, src_lang=cure_lang)
