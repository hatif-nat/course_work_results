import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics.cluster import adjusted_rand_score
import statistics
from sklearn.preprocessing import LabelEncoder

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.multiclass  import unique_labels

import nltk
nltk.download('wordnet')
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer 
import pandas as pd
import re

import csv
from easynmt import EasyNMT


model = EasyNMT('mbart50_m2en')

f = open("train.csv")
reader = csv.DictReader(f, delimiter='\t')
all_info = []
strings = []
for row in reader:
    all_info.append(row)
    strings.append(row["context"])
	
all_info = pd.read_csv('train.csv',  delimiter='\t')


f2 = open('trans_mbart.txt', 'w')
print("\nTranslating base mBart:")
translations2 = model.translate(strings, target_lang='en')
print("\nTranslating end:")
for sent, trans in zip(strings, translations2):
	f2.write(sent + " ||| " +  trans + '\n')
f2.close()


strings_only_target_sen = list(all_info.context)
all_info['only_target_sen_pos'] = all_info.positions
for i, string in enumerate(strings_only_target_sen):
	first_pos, second_pos = re.split('-', all_info.positions[i])
	word = string[int(first_pos): int(second_pos) + 1]
	sentences = re.split(r'(.*?[\.\!\?]{1}[\s]?)', string)
	sentences = list(filter(lambda a: a != '', sentences))
	for sentence in sentences:
		if word in sentence:
			strings_only_target_sen[i] = sentence
			break
	new_pos = strings_only_target_sen[i].find(word)
	all_info.only_target_sen_pos[i] = str(new_pos) + '-' + str(new_pos+ len(word) - 1)
  #print(strings_only_target_sen[i])

f2 = open('only_target_sen_trans_mbart.txt', 'w')
translations = model.translate(strings_only_target_sen, target_lang='en')
print("\nTranslations:")

for sent, trans in zip(strings_only_target_sen, translations):
	f2.write(sent + " ||| " +  trans + '\n')
f2.close()





