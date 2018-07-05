from bs4 import BeautifulSoup

def extract(tr):
    """
    effect      - return :: name, effect, constituent
    cure        - return :: name, cure, constituent
    constituent - return :: name, constituent, cure
    all         - return :: name, constituent, cure
    """
    td_list = tr.find_all('td')
    if len(td_list) < 4:
        print(tr)
        return None, None, None
    return td_list[1].string,td_list[2].string,td_list[3].string

def parse(f_name, out_name):
    html = BeautifulSoup(open(f_name),'html.parser')
    print('finished parsing html')
    write = open(out_name, 'w')
    tr_list = html.find_all('tr')
    print('number of prescriptions',len(tr_list)-1)
    pair_num = 0
    no_name_num = 0
    for tr in tr_list[1:]:
        name,constituent,effect = extract(tr)
        if constituent:
            write.write((str(constituent.replace('\n',''))+'\n'))
        '''
        if constituent and effect:
            pair_num += 1
            if not name:
                no_name_num += 1
            else:
                #write.write((str(name.replace('\n',''))+'\t\t'+str(constituent.replace('\n',''))+'\t\t'+str(effect.replace('\n',''))+'\n'))
        else:
            print(tr)
        '''
    write.close()
    print('pair number', pair_num)
    print('no name number', no_name_num)

if __name__ == '__main__':
    #parse('effect.html', 'effect.tab')
    #parse('cure.html', 'cure.tab')
    parse('constituent.html', 'constituent.tab')
