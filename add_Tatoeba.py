import pandas as pd

extra_data = pd.read_csv('extraWords.tsv', delimiter = '\t', quoting = 3)
extra_data.columns = ['num', 'rus', 'num2', 'en']

f3 = open('trans_mbart_extra.txt', 'w')
f = open('trans_mbart.txt', 'r')
a = f.readline()
while (a != ''):
	f3.write(a)
	a = f.readline()
for i in range(extra_data.shape[0]):
	f3.write(extra_data.rus[i] + " ||| " + extra_data.en[i] + '\n' )
f.close()
f3.close()

f3 = open('only_target_sen_trans_mbart_extra.txt', 'w')
f = open('only_target_sen_trans_mbart.txt', 'r')
a = f.readline()
while (a != ''):
	f3.write(a)
	a = f.readline()
for i in range(extra_data.shape[0]):
	f3.write(extra_data.rus[i] + " ||| " + extra_data.en[i] + '\n' )
f.close()
f3.close()