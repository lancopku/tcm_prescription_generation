import seq2seq_weak
import sys
import numpy as np
from tqdm import tqdm
import torch 
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import pickle, time
import torch
from language import Lang
import random
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

use_cuda = torch.cuda.is_available()
MAX_LENGTH = 20

class Decoder(nn.Module):
    def __init__(self, args, voc_size, emb_size, hidden_size, bidirectional=False):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.voc_size = voc_size
        if bidirectional:
            self.out = nn.Linear(self.hidden_size*2, self.voc_size)
        else:
            self.out = nn.Linear(self.hidden_size, self.voc_size)

    def forward(self, hidden):
        pred_word = self.out(hidden)
        return pred_word


class Encoder(nn.Module):
    def __init__(self, voc_size, emb_size, hidden_size, n_layers=1, bidirectional=False):
        super(Encoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.bidirectional = bidirectional
        
        self.embedding = nn.Embedding(voc_size, emb_size, padding_idx=0)
        self.gru = nn.GRU(emb_size, hidden_size, num_layers=self.n_layers, batch_first=False,
                          bidirectional=self.bidirectional)
    
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
        mask = torch.gt(input.data, 0)
        input_length = torch.sum((mask.long()), dim=1)  # batch first = True, (batch, sl)
        lengths, indices = torch.sort(input_length, dim=0, descending=True)
        _, ind = torch.sort(indices, dim=0)
        input_length = torch.unbind(lengths, dim=0)
        embedded = self.embedding(torch.index_select(input, dim=0, index=Variable(indices)))
        output, hidden = self.gru(pack(embedded, input_length, batch_first=True), hidden)
        output = torch.index_select(unpack(output, batch_first=True)[0], dim=0, index=Variable(ind)) * Variable(
            torch.unsqueeze(mask.float(), -1))
        hidden = torch.unbind(hidden, dim=0)
        hidden = torch.cat(hidden, 1)
        hidden = torch.index_select(hidden, dim=0, index=Variable(ind))
        direction = 2 if self.bidirectional else 1
        assert hidden.size() == (input.size()[0], self.hidden_size*direction) and output.size() == (
        input.size()[0], input.size()[1], self.hidden_size * direction)
        return output, hidden
    
    def initHidden(self, batch_size):
        bid = 2 if self.bidirectional else 1
        result = Variable(torch.zeros(self.n_layers * bid, batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda()
        return result

def target_padding(targets,length,pad_token=-1):
    result = [list() for i in range(len(targets))]
    for i in range(len(targets)):
        for j in range(len(targets[i])):
            if targets[i][j] < 1:
                break
            result[i].append(targets[i][j])
        while len(result[i]) < length:
            result[i].append(pad_token)
    return result

def multi_loss(logits, targets):
    '''
        logits :: bs, nv
        targets:: nclass, bs
    '''
    loss_func = torch.nn.MultiLabelMarginLoss()
    '''
    p = Variable(multi_hot(targets.data,logits.size()[1]))   # bs, nv
    if use_cuda:
        p = p.cuda()
    '''
    loss = loss_func(input=F.sigmoid(logits),target=targets)
    return loss

def train(args, inputs, targets, encoder, decoder, optimizer, tgt_lang):
    optimizer.zero_grad()
    input_length = inputs.size()[1]
    target_length = targets.size()[1]
    batch_size = inputs.size()[0]
    loss = 0.

    # encoding part
    encoder_output, encoder_state = encoder(inputs)

    # decoding part
    pred_words = decoder(encoder_state)     # bs, nv
    loss = multi_loss(pred_words, targets)
    loss.backward()
    optimizer.step()

    return loss.data

def evaluate(args, encoder, decoder, sentence, src_lang, tgt_lang, from_text=False, max_length=MAX_LENGTH):
    input_variable = seq2seq_weak.variable_from_sentence(src_lang,sentence, from_text)
    input_variable = torch.unsqueeze(input_variable, 0)
    input_length = input_variable.size()[1]
    encoder_output, encoder_state = encoder(input_variable)

    decoded_words = F.sigmoid(decoder(encoder_state))
    decoded_words = torch.squeeze(decoded_words)
    topv,topi = decoded_words.topk(max_length)
    results = []
    for v,i in zip(topv,topi):
        if v.data[0] > args.threshold:
            results.append(tgt_lang.id2word[i.data[0]])
    return results
    
def evaluate_all(args, data, encoder, decoder, src_lang, tgt_lang, from_text):
    hypothsis = []
    reference = []
    for s,t in tqdm(data, disable=not args.verbose):
        result = evaluate(args, encoder, decoder, list(s), src_lang=src_lang,tgt_lang=tgt_lang, from_text=from_text)
        hypothsis.append(result)
        if not from_text:
            reference.append([tgt_lang.id2word[word] for word in t])
        else:
            reference.append(t)
    precision, recall, F = seq2seq_weak.eval_F(reference, hypothsis, log_path='./log/'+args.log_dir)
    return precision, recall, F

def train_iters(args,encoder, decoder, train_data, dev_data, test_data, test_data_text, n_iters, src_lang, tgt_lang, print_every=1000, plot_every=1000):
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
            tgt = target_padding(tgt,len(tgt_lang.word2id))
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

if __name__ == '__main__':
    args = seq2seq_weak.parse_args()
    if use_cuda: torch.cuda.set_device(args.gpu)
    data, test_data_text, med_lang, cure_lang = pickle.load(open('../dataset/prescription_pairs_data.pkl','rb'))
    encoder = Encoder(voc_size=len(cure_lang.word2id), emb_size=args.emb_size, hidden_size=args.hidden_size, bidirectional=args.bidirectional)
    decoder = Decoder(args=args, voc_size=len(med_lang.word2id), hidden_size=args.hidden_size, emb_size=args.emb_size, bidirectional=args.bidirectional)
    encoder.load_emb(cure_lang.emb)
    if use_cuda:
        encoder.cuda()
        decoder.cuda()
    random.seed(255)
    random.shuffle(data)
    train_data = data[:int(len(data)*0.9)]
    dev_data = data[int(len(data)*0.9):int(len(data)*0.95)]
    test_data = data[int(len(data)*0.95):]
    batches = seq2seq_weak.make_batches(train_data, args.batch_size,med_lang)
    train_iters(args,encoder, decoder, batches, dev_data, test_data, test_data_text, args.epoch_num, tgt_lang=med_lang, src_lang=cure_lang)
