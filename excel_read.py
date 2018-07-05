import xlrd
import sys
import pickle

fname = sys.argv[1]
book = xlrd.open_workbook(fname)
sheet = book.sheets()[0]
med_map = {}
for i in range(sheet.nrows):
    line = sheet.row_values(i)
    if len(line) == 2:
        m1,m2 = line
        if len(m2) > 0:
            m1 = m1.split()[0]
            med_map[m1] = m2
            print(m1)
pickle.dump(med_map,open('med_map.pkl','wb'))
