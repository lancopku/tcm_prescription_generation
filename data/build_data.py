from language import Lang
import random
import pickle
import numpy as np
from tqdm import tqdm


med_map = pickle.load(open('med_map.pkl','rb'))     # manual projection in excel, written in excel_read.py

def process(constituent):
    pat = re.compile(u'(（.+?）)|(\\(.+?\\))|(第[0-9一二三四五六七八九十]+日)|(各?((等分)|(少许)|(减半)|(不拘多少)))')     # remove brackets, either in English or in Chinese
    meds = re.sub(pat,' ',constituent)
    sep = re.compile(u'、|，|,|:|：|；')
    meds = re.sub(sep,' ',meds)
    meds = meds.split(u'。')
    meds = meds[0].split(' ')
    return [normalize(med) for med in meds]

class Prescription:
    def __init__(self, name):
        self.name = name
        self.effect = None
        self.cure = None
        self.constituent = None

    def parse_constituent(self):
        self.constitent_list = process(self.constituent)

def read_vec(fname='char_vec.txt',vec_len = 100):
    vec = {}
    for l in open(fname):
        tem = l.strip().split()
        if len(tem) < vec_len+1:      # the length of a vector
            continue
        word = tem[0]
        v = np.array([float(num) for num in tem[1:]])
        vec[word] = v
    return vec

def cons2vec(constituents, med_vec, lang):
    if lang.emb == None:
        lang.init_emb()
        lang.use_pre_emb = True
    result = []
    for med in constituents:
        if med in med_vec:
            if med not in lang.word2id:
                lang.add_word(med)
                lang.emb.append(med_vec[med])
            result.append(lang.word2id[med])
        '''
        else:
            result.append(lang.word2id['OOV'])
        '''
    result.append(lang.EOS_token)
    return result
        
def text2vec(sentence, char_vec, lang):
    if not sentence:
        return None
    if lang.emb == None:
        lang.init_emb()
        lang.use_pre_emb = True
    result = []
    for c in sentence:
        if c in char_vec:
            if c not in lang.word2id:
                lang.add_word(c)
                lang.emb.append(char_vec[c])
            result.append(lang.word2id[c])
        else:
            result.append(lang.word2id['OOV'])
    result.append(lang.EOS_token)
    return result

def read_tab(fname, prescriptions, mode='cure'):
    name_constituents = {}
    for line in open(fname):
        tem = line.strip().split('\t\t')
        if len(tem) < 3:
            print(line)
            continue
        name, mid, constituent = tem
        name_constituents[name] = constituent
        prescriptions.append(Prescription(name))
        prescriptions[-1].constituent = constituent
        if mode == 'cure':
            prescriptions[-1].cure = mid
        elif mode == 'effect':
            prescriptions[-1].effect = mid
    return name_constituents

def read_test_data(fname, char_vec, med_vec, cure_lang, med_lang):
    oov_num = 0
    f = open(fname).readlines()
    data = []
    for l in f:
        if len(l.strip().split('\t'))<2:
            print(l)
            continue
        src = l.split('\t')[0]
        tgt = l.strip().split('\t')[1].split()
        tgt_mapped = []
        for word in tgt:
            if word not in med_lang.word2id and word not in med_map:
                oov_num += 1
                print(word)
            if word in med_map:
                tgt_mapped.append(med_map[word])
            else:
                tgt_mapped.append(word)
        data.append((src, tgt_mapped))
    print('oov num',oov_num)
    return data
        
def sort_data(data):
    return sorted(data, key=lambda k:len(k[0]),reverse=True)

def detect_name(constituent, name_constituents):
    for name in name_constituents:
        if name in constituent:
            constituent = constituent.replace(name, ' '+name_constituents[name]+' ')
    return constituent

if __name__ == '__main__':
    prescriptions = []
    max_length = 0
    name_constituents = read_tab('cure.tab', prescriptions, 'cure')
    data = []
    med_lang = Lang('med')
    cure_lang = Lang('cure')
    char_vec = read_vec('char_vec.txt')
    med_vec = read_vec('medicine_norm_vec.txt')
    for p in tqdm(prescriptions):
        p.constituent = detect_name(p.constituent, name_constituents)
        p.parse_constituent()
        cons_vec = cons2vec(p.constitent_list, med_vec, med_lang)
        max_length = max(max_length, len(cons_vec))
        cure_vec = text2vec(p.cure, char_vec, cure_lang)
        if cure_vec:
            data.append((cure_vec, cons_vec))
    test_data_text = read_test_data('text_book.tab',char_vec, med_vec, cure_lang, med_lang)
    print('max length',max_length)
    # med_lang and cure_lang have corresponding embeddings
    pickle.dump((data, test_data_text, med_lang, cure_lang),open('prescription_pairs_data.pkl','wb'))
