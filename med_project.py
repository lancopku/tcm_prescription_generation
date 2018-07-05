import pickle
import json
import editdistance
from collections import Counter

voc = {}
for l in open('medicine_full_voc.txt'):
    word, num = l.decode('utf8').strip().split()
    num = int(num)
    if num > 3:
        voc[word] = num
    
def get_similar_word(word):
    global voc
    min_count = len(word)
    sim_word_list = []
    for med in voc:
        if med in word or word in med:
            sim_word_list.append((med, 0))
            min_count = 0
        else:
            d = int(editdistance.eval(med,word))
            if d == min_count:
                sim_word_list.append((med, d))
            elif d < min_count:
                sim_word_list = [(med,d)]
                min_count = d
    assert len(sim_word_list) > 0
    max_word_count = (None,0)
    for term in sim_word_list:
        med, d = term
        if voc[med] > max_word_count[1]:
            max_word_count = (med, voc[med])
    if int(editdistance.eval(max_word_count[0],word)) >= len(max_word_count[0]):
        return None
    return max_word_count[0]
        
if __name__ == '__main__':
    write = open('fangji_norm.txt','w')
    med_map = {}
    for l in open('fangji.txt'):
        words = l.decode('utf8').strip().split()
        sentence = []
        for word in words:
            if word in voc:
                sentence.append(word)
            else:
                norm_word = get_similar_word(word)
                if norm_word != None:
                    med_map[word] = norm_word
                    sentence.append(norm_word)
                else:
                    print word.encode('utf8')
                    continue
        write.write((' '.join(sentence)).encode('utf8')+'\n')
    write.close()
    write = open('med_map.csv','w')
    for med in med_map:
        write.write(med.encode('utf8')+','+med_map[med].encode('utf8')+'\n')
    write.close()

