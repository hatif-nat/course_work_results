import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics.cluster import adjusted_rand_score
import statistics
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

import seaborn as sns
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.multiclass  import unique_labels

import nltk
#nltk.download('wordnet')
#nltk.download('punkt')
from nltk.stem import WordNetLemmatizer 
import pandas as pd
import re

import csv
from easynmt import EasyNMT

"""# Read examples"""

all_info = pd.read_csv('train.csv',  delimiter='\t')
f = open("train.csv")
reader = csv.DictReader(f, delimiter='\t')

strings = []
for row in reader:
    strings.append(row["context"])

set_words_2 = sorted(list(set(all_info.word)))

cool_words = ['клетка', 'среда', 'рысь', 'крыло', 'пропасть']

senses_of_words = pd.read_csv('SenseOfTrainWordsBTS.csv')

translates = {}
for word in set_words_2:
  translates[word] = {}
  
translates['балка']['1'] = ['	crossmember', 'truss', 'baulk', 'balk' , 'beam', 'summer', 'brace' , 'carrier', 'joist', 'peace', 'balka' , 'bar', 'ball']
translates['балка']['2'] = ['wash', 'draw', 'gill', 'cleavage', 'ravine', 'gullet']

translates['вид']['1'] =  ['guise', 'appearance','kind','form']
translates['вид']['2'] =  ['view', 'scene', ]
translates['вид']['5'] =  ['kind', 'sort', 'genus', 'species', 'type', 'format', 'aspect', 'mode', 'brand', 'state', 'style', 'hue', 'gender', 'category', ]
 
translates['винт']['1'] =  ['screw', 'bolt', 'stud']
translates['винт']['2'] =  ['propeller', 'rotor', 'helix']
translates['винт']['3'] =  ['stopper', 'plug', 'bibb', 'bung','cap' ]
translates['винт']['5'] =  ['vint']
 
translates['горн']['1'] =  ['furnace','hearth','forge','crucible']
translates['горн']['2'] = ['receiver', 'furnace']
translates['горн']['3'] = ['bugle','horn']
 
translates['губа']['1'] =  ['lip', 'labium', 'labrum']
translates['губа']['3'] =  ['bay','gulf','estuary','firth','channel']
translates['губа']['4'] =  ['guardroom','clink','calaboose']
 
translates['жаба']['1'] =  ['toad', 'frog', 'hoptoad', 'anuran']
translates['жаба']['2'] =  ['fish-face', 'gorgon', 'hag', 'freak', 'ugly']
translates['жаба']['3'] =  ['greed', 'avarice', 'greediness','covetousness','avidity','avidity', 'rapacity','stinginess','gimmies', ]
translates['жаба']['4'] =  ['quinsy', 'stenocardia', 'angina']
 
translates['клетка']['1'] =  ['cell','cage','hutch','mew','birdcage']
translates['клетка']['2'] =  ['clamp','pane' ]
translates['клетка']['3'] =  ['box', 'square', 'quadrangle', 'quadrilateral', 'rectangle']
translates['клетка']['4'] =  ['cell', 'cellula', 'membrane']
translates['клетка']['5'] =  ['staircase', 'stairwell', 'stairway']
translates['клетка']['6'] = ['chest', 'ribcage', 'cage']
 
translates['купюра']['1'] =  ['cut', 'abridgement', 'redacted', 'note']
translates['купюра']['2'] =  ['bill', 'banknote','omission','money','currency' ]
 
translates['курица']['1'] =  ['chicken', 'hen', 'fowl']
translates['курица']['2'] =  ['chicken', 'meat']
 
translates['лавка']['1'] =  ['bench', 'seat', 'pew', 'stoo;']
translates['лавка']['2'] =  ['shop','store', 'stall', 'grocery', 'canteen', 'trade', 'boutique']
 
translates['лайка']['1'] = ['husky', 'laika','malamute','wolfhound']
translates['лайка']['2'] = ['chevrette', 'dog-skin', 'dogskin', 'kidskin', 'leather','glace']
 
translates['лев']['1'] =  ['loin', 'cat']
translates['лев']['2'] =  ['brave', 'valiant','courageous', 'bold', 'gallant','trojan' ]
translates['лев']['3'] =  ['trendsetter', 'setter', 'bellwether', 'tastemaker', 'influencers']
translates['лев']['4'] =  ['leo', 'lion']
 
translates['лира']['1'] =  ['lyre']
translates['лира']['2'] =  ['lira']
 
translates['мина']['1'] =  ['mine', 'bomb', 'shell', 'landmine','projectile', 'torpedo']
translates['мина']['2'] =  ['look', 'air', 'face','expression', 'countenance']
translates['мина']['3'] =  ['frame','relay']
 
translates['мишень']['1'] =  ['target', 'aim', 'mark','bullseye', 'goal', 'objective']
translates['мишень']['2'] =  ['target', 'aim','cockshy', 'butt','sitter']

translates['крыло']['1'] = ['wing']
translates['крыло']['2'] = ['wing', 'aerofoil', 'airfoil', 'plane']
translates['крыло']['3'] = ['vane', ]
translates['крыло']['4'] = ['splashboard']
translates['крыло']['5'] = ['wing']
translates['крыло']['6'] = ['annexe', 'flanker', 'annex']
translates['крыло']['7'] = ['flank']
translates['крыло']['8'] = ['wing']

translates['обед']['1'] = ['dinner', 'luncheon', 'lunch', 'potluck']
translates['обед']['2'] = ['meal', 'lunch', 'repast']
translates['обед']['3'] = ['lunchtime', 'noon', 'afternoon']
translates['обед']['4'] = ['beantime', 'break']

translates['оклад']['1'] = ['salary', 'wage', 'raise', 'fee', 'pay', 'remuneration' ,'emolument', 'stipend']
translates['оклад']['2'] = ['assessment']
translates['оклад']['3'] = ['cover', 'framework']

translates['опушка']['1'] = ['edge', 'clearing', 'skirt', 'tree', 'margin', 'marge', 'fringe', 'glade', 'border']
translates['опушка']['2'] = ['trimming', 'trim', 'fur']

translates['полис']['1'] = ['state', 'city', 'polis']
translates['полис']['2'] = ['policy']

translates['пост']['1'] = ['outpost', 'station']
translates['пост']['2'] = ['post', 'position', 'place', 'office', 'seat']
translates['пост']['3'] = ['fasting', 'ramadan']
translates['пост']['4'] = ['post', 'message']
translates['пост']['5'] = ['post']

translates['поток']['1'] = ['stream', 'flood' , 'deluge', 'spill', 'grush', 'river', 'outpour', 'tide', 'gout' ]
translates['поток']['2'] = ['flow', 'flux', 'grush']

translates['проказа']['1'] = ['prank', 'lark', 'laverock', 'caper']
translates['проказа']['2'] = ['leprosy', 'mischief'  ]

translates['пропасть']['1'] = ['abyss', 'precipice', 'chasm', 'pit', 'gulf', 'abysm',]
translates['пропасть']['2'] = ['gap', 'chaos']
translates['пропасть']['3'] = ['lot', 'most', 'many', 'much' ]
translates['пропасть']['4'] = ['annoyance', 'vexation', 'disappointment', 'chagrin', 'exasperation', 'frustration', 'anger']

translates['проспект']['1'] = ['prospekt']
translates['проспект']['2'] = ['summary', 'resume', 'outline', 'recap', 'headlines']
translates['проспект']['3'] = ['brochure', 'pamphlet', 'booklet', 'folder', 'circular']
translates['проспект']['4'] = ['avenue', 'street', 'boulevard' ]

translates['пытка']['1'] = ['torture', 'excruciation',  'inquisition', 'agony', 'torment',  ]
translates['пытка']['2'] = ['agony', 'question','torment' , 'torture']

translates['рысь']['1'] = ['troat']
translates['рысь']['2'] = ['lynx', 'ounce', 'bobcat']

translates['среда']['1'] = ['medium', 'environment', 'media' ]
translates['среда']['2'] = ['circumstance', 'environment', 'ambience',  'surroundings', 'milieu', 'milieu']
translates['среда']['3'] = ['circumstance', 'environment', 'field', 'ambience',  'surroundings', 'milieu', 'community' , 'milieu']
translates['среда']['4'] = ['wednesday', 'mid-week', 'Wednesday']

translates['штамп']['1'] = ['stamp', 'imprint', 'print', 'postmark', 'impress', 'signet']
translates['штамп']['2'] = ['die', 'punch']
translates['штамп']['3'] = ['seal', 'press']
translates['штамп']['4'] = ['cliche', 'pattern', 'trope ']

translates['хвост']['1'] = ['tail', 'brush', 'bob', 'flag', 'stern', 'train', 'rudder', 'scut', 'stern', 'cue']
translates['хвост']['3'] = ['tail']
translates['хвост']['4'] = ['tail', 'back', 'shadow', 'behind', 'last', 'left', 'along']
translates['хвост']['7'] = ['debt', 'credit']

"""# Functions

found_translate_word - находит сопоставленное слово (словосочетание) для многозначного слова из предложения
"""

def found_translate_word(positions, align, str_align):
    first_pos, second_pos = re.split('-', positions)
    str1, str2 = re.split('[ ][|]{3}[ ]', str_align)
    #print(str1)
    l1 = re.split('[\t\ ]', str1)
    #l1 = re.split('[\s.,!?\(\)\[\]\{\}\'\"\<\>;:«»]+', str1)
    #print(l1)
    #l1[:] =[re.sub(r'\W?([А-Яа-я]+)', r'\1', el) if re.fullmatch(r'\W?([А-Яа-я]+)', el) else el for el in l1 ]
    #l1[:] =[re.sub(r'([А-Яа-я]*)\W?', r'\1', el) if re.fullmatch(r'([А-Яа-я]+)\W+', el) else el for el in l1 ]
    l2 = re.split('[\t\ ]', str2)
    #l2 = re.split('[\s.,!?\(\)\[\]\{\}\'\"\<\>;:«»]+', str2)
    #l2[:] =[re.sub(r'([A-Za-z]*)\W?', r'\1', el) if re.fullmatch(r'([A-Za-z]*)\W?', el) else el for el in l2]
    #l2[:] =[re.sub(r'\W?([A-Za-z]*)', r'\1', el) if re.fullmatch(r'\W?([A-Za-z]*)', el) else el for el in l2]
    
    word = str1[int(first_pos): int(second_pos) + 1]
    #print(word)
    #print(l1)
    for i, item in enumerate(l1):
        if word in item:
            x = i
            break
    alls = re.split(' ', align)
    ret = ''
    for al in alls:
        a, b = re.split('[-]', al)
        if (int(a) == x):
            ret += (l2[int(b)]) + ' '
    return ret

"""countTranslates - составляет словарь из слов, где для каждого слова определены номера правильных значений, и для каждого значения подсчитано, сколько и какие слова (словосоченания) были сопоставлены с многозначным словом, а также список из сопоставленных переводов многозначных слов"""

def countTranslates(positions, aligns, str_aligns):
    count_trans = {}
    words = []
    for i, it in enumerate(trans):
        a = found_translate_word(positions[i], aligns[i], str_aligns[i])
        if re.fullmatch(r'\W*(\w+)\W+', a):
            a = re.sub(r'\W*(\w+)\W+', r'\1', a)
        if re.fullmatch(r'\W*(\w+)\W*(\s)\W*(\w+)\W+', a):
            a = re.sub(r'\W*(\w+)\W*(\s)\W*(\w+)\W+', r'\1\2\3', a)
        lemmatizer = WordNetLemmatizer()
        word_list = nltk.word_tokenize(a)
        a = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
        a = a.lower()
        words.append(a)
        if (all_info.word[i]) not in count_trans:
            count_trans[all_info.word[i]] = {}
            count_trans[all_info.word[i]][all_info.gold_sense_id[i]] = {}
            count_trans[all_info.word[i]][all_info.gold_sense_id[i]][a] = 1
        elif all_info.gold_sense_id[i] not in count_trans[all_info.word[i]]:
            count_trans[all_info.word[i]][all_info.gold_sense_id[i]] = {}
            count_trans[all_info.word[i]][all_info.gold_sense_id[i]][a] = 1
        elif a not in count_trans[all_info.word[i]][all_info.gold_sense_id[i]]:
            count_trans[all_info.word[i]][all_info.gold_sense_id[i]][a] = 1
        else:
            count_trans[all_info.word[i]][all_info.gold_sense_id[i]][a] += 1
        #print(i+1, all_info.word[i], all_info.gold_sense_id[i], a)   
    #print(count_trans)
    return count_trans, words

""" CoolInfo - return ARI for the transmitted word"""

def CoolInfo(word):
  #print(word)
  #print heat map

  #print confusion matrix doesnt work

  num_examples = [-1, -1]
  for i in range(all_info.shape[0]):
    if all_info.word[i] == word:
      if num_examples[0] == -1:
        num_examples[0] = i
        num_examples[1] = i
      num_examples[1] += 1
  true_senses = []
  predict_senses = []
  num_word_senses = {}
  for i in range(senses_of_words.shape[0]):
    if word == senses_of_words.word[i]:
      num_word_senses[senses_of_words.gold_sense_id[i]] = senses_of_words.sense[i]
  for i in range(num_examples[0], num_examples[1]):
    true_senses.append(num_word_senses[all_info.gold_sense_id[i]])
    predict_senses.append(al_words[i])

  #ARI

  set_words = {}
  count = 0
  predict_clusters = []
  for word in al_words:
    if word in set_words:
      predict_clusters.append(set_words[word])
    else:
      set_words[word] = count
      count += 1
      predict_clusters.append(set_words[word])

  true_clasters = []
  for sen in all_info.gold_sense_id:
    true_clasters.append(sen - 1)

  word_predict_clusters = predict_clusters[num_examples[0]: num_examples[1]]
  word_true_clusters = true_clasters[num_examples[0]: num_examples[1]]

  cm = confusion_matrix(word_true_clusters, word_predict_clusters, sample_weight=None,
                          labels=None, normalize=None)

  display_labels = unique_labels(word_true_clusters, word_predict_clusters)


  disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=display_labels)

  a = (format(adjusted_rand_score(word_true_clusters, word_predict_clusters), ".4f"))
  return a
  #return disp.plot(include_values=True, cmap='viridis', ax=None, xticks_rotation='horizontal',values_format=None)

"""clasterize - проходится по каждой паре слов, и если их длины больше 3х (чтобы пропустить все артикли и предлоги) и одно полностью состоит в другом (чтобы объединять словосочетания 'the toad' и 'toad')"""

def clasterize(al_words):
  for i in range(len(al_words)):
    for j in range(i,  len(al_words)):
      #print(i, j, al_words[i], al_words[j])
      if len(al_words[i]) > 3 and len(al_words[j]) > 3:
        if (al_words[i] in al_words[j]):
          #print(al_words[i], al_words[j])
          al_words[j] = al_words[i]
        if (al_words[j] in al_words[i]):
          #print(al_words[i], al_words[j])
          al_words[i] = al_words[j]
  return al_words

def assoc(al_words):
  c_words = {}
  ma = []
  for strr in al_words:
    l2 = re.split('[\t\ ]', strr)
    ma.append(l2)
    if l2 != '':
      for w in l2:
        if w in c_words:
          c_words[w] += 1
        else :
          c_words[w] = 1
  for i in range(len(al_words)):
    best_word = ''
    best_res = 0
    if ma[i] != ['']:
      for el in ma[i]:
        if (c_words[el] > best_res):
          best_word = el
          best_res  = c_words[el]
    al_words[i] = best_word
  return al_words

def unific(al_wordss, transs):
  for i in range(len(al_wordss)):
    str1, str2 = re.split('[ ][|]{3}[ ]', transs[i])
    l2 = re.split('[\s]', str2)
    wor = all_info.word[i]
    end = 0
    for w in l2:    
      lemmatizer = WordNetLemmatizer()
      w2 = lemmatizer.lemmatize(w)
      w2 = re.sub(r'[^-a-zA-Z]', r'', w2)
      w2 = w2.lower()
      for j in translates[wor]:
        if w2 in translates[wor][j]:
          al_wordss[i] = w2
          end = 1
          break
      if end == 1:
        break
  return al_wordss

"""countTranslates2 - составляет словарь из слов, где для каждого слова определены номера правильных значений, и для каждого значения подсчитано, сколько и какие слова (словосоченания) были сопоставлены с многозначным словом (тоже самое, что и countTranslates, но без списка сопоставленных переводов многозначных слов)"""

def countTranslates2(al_words):
    count_trans = {}
    for i in range(len(al_words)):
        a = al_words[i]
        if (all_info.word[i]) not in count_trans:
            count_trans[all_info.word[i]] = {}
            count_trans[all_info.word[i]][all_info.gold_sense_id[i]] = {}
            count_trans[all_info.word[i]][all_info.gold_sense_id[i]][a] = 1
        elif all_info.gold_sense_id[i] not in count_trans[all_info.word[i]]:
            count_trans[all_info.word[i]][all_info.gold_sense_id[i]] = {}
            count_trans[all_info.word[i]][all_info.gold_sense_id[i]][a] = 1
        elif a not in count_trans[all_info.word[i]][all_info.gold_sense_id[i]]:
            count_trans[all_info.word[i]][all_info.gold_sense_id[i]][a] = 1
        else:
            count_trans[all_info.word[i]][all_info.gold_sense_id[i]][a] += 1
    return count_trans

def print_conf_matrix(al_words, word, num = 8, name = ''):
  num_examples = [-1, -1]
  for i in range(all_info.shape[0]):
    if all_info.word[i] == word:
      if num_examples[0] == -1:
        num_examples[0] = i
        num_examples[1] = i
      num_examples[1] += 1
  y_pred = al_words[num_examples[0]: num_examples[1]]
  y_true = all_info.gold_sense_id[num_examples[0]: num_examples[1]]
  dic = {}
  for i in range(len(senses_of_words)):
    if senses_of_words.word[i] == word:
      dic[senses_of_words.gold_sense_id[i]] = str(senses_of_words.gold_sense_id[i]) + ': ' + senses_of_words.sense[i]
  y_true = [dic[x] for x in y_true]
  pred_encoder = LabelEncoder().fit(y_pred)
  y_pred_enc = pred_encoder.transform(y_pred)

  true_encoder = LabelEncoder().fit(y_true)
  y_true_enc = true_encoder.transform(y_true)

  conf_matr = confusion_matrix(y_true_enc, y_pred_enc,)
  conf_matr = conf_matr[:len(true_encoder.classes_)][:len(pred_encoder.classes_)]

  ticks=np.arange(conf_matr.min(),conf_matr.max()+1 )
  boundaries = np.arange(conf_matr.min()-.5,conf_matr.max()+1.5 )
  cmap = plt.get_cmap("YlGnBu", conf_matr.max()-conf_matr.min()+1)

  plt.figure(figsize = (12,num))

  ax = sns.heatmap(conf_matr, annot=True, fmt="d", xticklabels = pred_encoder.classes_,  
                  yticklabels = true_encoder.classes_, linewidths=0.4, cmap=cmap, 
                  cbar_kws={"ticks":ticks, "boundaries":boundaries})
  ax.set(xlabel='pred_sense_id', ylabel='gold_sense_id', title = (word + ' ' + name))
  #plt.show()

  plt.savefig(word + name + ".pdf", bbox_inches='tight')

"""Так как из awesome-align сопоставления приходят в странном порядке, то их нужно отсортировать : 0-0 0-1 1-2 2-1 и т д"""

def sort_align(stringg):
    tmp_array = [tuple([int(x) for x in pair.split('-')]) for pair in stringg.split()]
    tmp_array.sort()
    ans_string = ' '.join(['{}-{}'.format(a,b) for a,b in tmp_array])
    return ans_string



"""Заметим, что лучше всего работает комбинация: с добавлением Tatoeba и дополнительным объединением слов - сравним базовую версию других переводчиков и с добавлением Tatoeba и дополнительным объединением слов

# Use another translate - mbart

Заметим, что в большинстве случаев целевое слово сопоставляется с самым частым его переводом, а значит переводчик не очень хорошо переводит. Попробуем использовать другой переводчик: mbart50_m2en
"""

"""#Try translate only target sentence

Предположим, что переводчики будут лучше переводить, если оставить только целевое предложение

Пересчитаем положение многозначного слова в предложении
"""
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


f4 = open("results2.txt", 'w')
"""# mBart + awesome-align"""

f2 = open("trans_mbart.txt", 'r')

trans = f2.read().split('\n')
f2.close()

trans.remove('') 

f = open('align_mbart_awesome', 'r')
aligns = []

for i in range(0, len(trans)):
    aligns.append(sort_align(f.readline()[:-1]))
    
f.close()

count_trans, al_words = countTranslates(all_info.positions, aligns, trans)

"""Tatoeba добавить уже не получится - слишком долго работает awesome-align. Но можно сопоставить"""

al_words = assoc(al_words)
count_trans = countTranslates2(al_words)

true_true = 0
true_false = 0
false_true = 0
false_false = 0
strange = 0
t = 0
for i in range(len(al_words)):
  if al_words[i] in translates[all_info.word[i]][str(all_info.gold_sense_id[i])]:
     true_true += 1
  else:
    t = 0
    for el in translates[all_info.word[i]]:
      if al_words[i] in translates[all_info.word[i]][el]:
        false_true += 1
        t = 1
        break
    if (t == 0):
          str1, str2 = re.split('[ ][|]{3}[ ]', trans[i])
          l2 = re.split('[\s]', str2)
          wor = all_info.word[i]
          end = 0
          for w in l2:    
            lemmatizer = WordNetLemmatizer()
            w2 = lemmatizer.lemmatize(w)
            w2 = re.sub(r'[^-a-zA-Z]', r'', w2)
            w2 = w2.lower()
            if w2 in translates[all_info.word[i]][str(all_info.gold_sense_id[i])]:
              true_false += 1
              end = 1
              break
            for j in translates[wor]:
              if w2 in translates[wor][j]:
                false_false += 1
                end = 1
                break
            if end == 1:
              break
          if end == 0:
            strange += 1     
      #print('strange', all_info.word[i], al_words[i])
print('\n\nResults of  base mBart + awesome-align + assoc:\n')
print('Доля правильных переводов и правильных сопоставлений:   ', format(true_true / len(al_words), '.5f'))
print('Доля правильных переводов и неправильных сопоставлений:   ', format(true_false / len(al_words), '.5f'))
print('Доля неправильных переводов и правильных сопоставлений: ', format(false_true / len(al_words), '.5f'))
print('Доля неправильных переводов и неправильных сопоставлений: ', format(false_false / len(al_words), '.5f'))
print('Доля странных сопоставлений и переводов:     ', format(strange / len(al_words), '.5f'))

f4.write('\n\nResults of  base mBart + awesome-align + assoc :\n')
f4.write('Доля правильных переводов и правильных сопоставлений:   '+ str( format(true_true / len(al_words), '.5f')) +  '\n')
f4.write('Доля правильных переводов и неправильных сопоставлений:   ' + str( format(true_false / len(al_words), '.5f')) + '\n')
f4.write('Доля неправильных переводов и правильных сопоставлений: ' + str( format(false_true / len(al_words), '.5f')) +'\n')
f4.write('Доля неправильных переводов и неправильных сопоставлений: ' + str( format(false_false / len(al_words), '.5f'))  +'\n')
f4.write('Доля странных сопоставлений и переводов:     '+  str(format(strange / len(al_words), '.5f')) + '\n')
print('\n')


al_words = unific(al_words, trans)
count_trans = countTranslates2(al_words)

true_true = 0
true_false = 0
false_true = 0
false_false = 0
strange = 0
t = 0
for i in range(len(al_words)):
  if al_words[i] in translates[all_info.word[i]][str(all_info.gold_sense_id[i])]:
     true_true += 1
  else:
    t = 0
    for el in translates[all_info.word[i]]:
      if al_words[i] in translates[all_info.word[i]][el]:
        false_true += 1
        t = 1
        break
    if (t == 0):
          str1, str2 = re.split('[ ][|]{3}[ ]', trans[i])
          l2 = re.split('[\s]', str2)
          wor = all_info.word[i]
          end = 0
          for w in l2:    
            lemmatizer = WordNetLemmatizer()
            w2 = lemmatizer.lemmatize(w)
            w2 = re.sub(r'[^-a-zA-Z]', r'', w2)
            w2 = w2.lower()
            if w2 in translates[all_info.word[i]][str(all_info.gold_sense_id[i])]:
              true_false += 1
              end = 1
              break
            for j in translates[wor]:
              if w2 in translates[wor][j]:
                false_false += 1
                end = 1
                break
            if end == 1:
              break
          if end == 0:
            strange += 1     
      #print('strange', all_info.word[i], al_words[i])
print('\n\nResults of  base mBart + awesome-align + assoc + unific:\n')
print('Доля правильных переводов и правильных сопоставлений:   ', format(true_true / len(al_words), '.5f'))
print('Доля правильных переводов и неправильных сопоставлений:   ', format(true_false / len(al_words), '.5f'))
print('Доля неправильных переводов и правильных сопоставлений: ', format(false_true / len(al_words), '.5f'))
print('Доля неправильных переводов и неправильных сопоставлений: ', format(false_false / len(al_words), '.5f'))
print('Доля странных сопоставлений и переводов:     ', format(strange / len(al_words), '.5f'))

f4.write('\n\nResults of  base mBart + awesome-align + assoc + unific:\n')
f4.write('Доля правильных переводов и правильных сопоставлений:   '+ str( format(true_true / len(al_words), '.5f'))  + '\n')
f4.write('Доля правильных переводов и неправильных сопоставлений:   ' + str( format(true_false / len(al_words), '.5f')) + '\n')
f4.write('Доля неправильных переводов и правильных сопоставлений: ' + str( format(false_true / len(al_words), '.5f')) + '\n')
f4.write('Доля неправильных переводов и неправильных сопоставлений: ' + str( format(false_false / len(al_words), '.5f')) + '\n')
f4.write('Доля странных сопоставлений и переводов:     '+  str(format(strange / len(al_words), '.5f')) + '\n')


print('\n')


"""# Only target sen + awesome-align

# mBart + only target sen + awesome-align
"""

f2 = open("only_target_sen_trans_mbart.txt", 'r')

trans = f2.read().split('\n')
f2.close()

trans.remove('') 

f = open('only_target_sen_trans_mbart_awesome', 'r')
aligns = []

for i in range(0, len(trans)):
    aligns.append(sort_align(f.readline()[:-1]))
    
f.close()

count_trans, al_words = countTranslates(all_info.only_target_sen_pos, aligns, trans)



al_words = clasterize(al_words)
al_words = unific(al_words, trans)
count_trans = countTranslates2(al_words)

"""Посмотрим, в какой части предложений многозначное слово правильно перевелось и сопоставилось со своим значением, с неправильным значением (то есть переводом другого значения данного слова), а также каков процент непонятных левых слов"""

true_true = 0
true_false = 0
false_true = 0
false_false = 0
strange = 0
t = 0
for i in range(len(al_words)):
  if al_words[i] in translates[all_info.word[i]][str(all_info.gold_sense_id[i])]:
     true_true += 1
  else:
    t = 0
    for el in translates[all_info.word[i]]:
      if al_words[i] in translates[all_info.word[i]][el]:
        false_true += 1
        t = 1
        break
    if (t == 0):
          str1, str2 = re.split('[ ][|]{3}[ ]', trans[i])
          l2 = re.split('[\s]', str2)
          wor = all_info.word[i]
          end = 0
          for w in l2:    
            lemmatizer = WordNetLemmatizer()
            w2 = lemmatizer.lemmatize(w)
            w2 = re.sub(r'[^-a-zA-Z]', r'', w2)
            w2 = w2.lower()
            if w2 in translates[all_info.word[i]][str(all_info.gold_sense_id[i])]:
              true_false += 1
              end = 1
              break
            for j in translates[wor]:
              if w2 in translates[wor][j]:
                false_false += 1
                end = 1
                break
            if end == 1:
              break
          if end == 0:
            strange += 1     
      #print('strange', all_info.word[i], al_words[i])
print('\n\nResults of  only target sen + mBart + awesome-align + clasterize + unific:\n')
print('Доля правильных переводов и правильных сопоставлений:   ', format(true_true / len(al_words), '.5f'))
print('Доля правильных переводов и неправильных сопоставлений:   ', format(true_false / len(al_words), '.5f'))
print('Доля неправильных переводов и правильных сопоставлений: ', format(false_true / len(al_words), '.5f'))
print('Доля неправильных переводов и неправильных сопоставлений: ', format(false_false / len(al_words), '.5f'))
print('Доля странных сопоставлений и переводов:     ', format(strange / len(al_words), '.5f'))

f4.write('\n\nResults of  only target sen + mBart + awesome-align + clasterize + unific:\n')
f4.write('Доля правильных переводов и правильных сопоставлений:   '+ str( format(true_true / len(al_words), '.5f')) + '\n')
f4.write('Доля правильных переводов и неправильных сопоставлений:   ' + str( format(true_false / len(al_words), '.5f')) + '\n')
f4.write('Доля неправильных переводов и правильных сопоставлений: ' + str( format(false_true / len(al_words), '.5f')) + '\n')
f4.write('Доля неправильных переводов и неправильных сопоставлений: ' + str( format(false_false / len(al_words), '.5f')) + '\n')
f4.write('Доля странных сопоставлений и переводов:     '+  str(format(strange / len(al_words), '.5f')) + '\n')



count_trans, al_words = countTranslates(all_info.only_target_sen_pos, aligns, trans)



al_words = assoc(al_words)
count_trans = countTranslates2(al_words)

"""Посмотрим, в какой части предложений многозначное слово правильно перевелось и сопоставилось со своим значением, с неправильным значением (то есть переводом другого значения данного слова), а также каков процент непонятных левых слов"""

true_true = 0
true_false = 0
false_true = 0
false_false = 0
strange = 0
t = 0
for i in range(len(al_words)):
  if al_words[i] in translates[all_info.word[i]][str(all_info.gold_sense_id[i])]:
     true_true += 1
  else:
    t = 0
    for el in translates[all_info.word[i]]:
      if al_words[i] in translates[all_info.word[i]][el]:
        false_true += 1
        t = 1
        break
    if (t == 0):
          str1, str2 = re.split('[ ][|]{3}[ ]', trans[i])
          l2 = re.split('[\s]', str2)
          wor = all_info.word[i]
          end = 0
          for w in l2:    
            lemmatizer = WordNetLemmatizer()
            w2 = lemmatizer.lemmatize(w)
            w2 = re.sub(r'[^-a-zA-Z]', r'', w2)
            w2 = w2.lower()
            if w2 in translates[all_info.word[i]][str(all_info.gold_sense_id[i])]:
              true_false += 1
              end = 1
              break
            for j in translates[wor]:
              if w2 in translates[wor][j]:
                false_false += 1
                end = 1
                break
            if end == 1:
              break
          if end == 0:
            strange += 1     
      #print('strange', all_info.word[i], al_words[i])
print('\n\nResults of  only target sen + mBart + awesome-align + assoc:\n')
print('Доля правильных переводов и правильных сопоставлений:   ', format(true_true / len(al_words), '.5f'))
print('Доля правильных переводов и неправильных сопоставлений:   ', format(true_false / len(al_words), '.5f'))
print('Доля неправильных переводов и правильных сопоставлений: ', format(false_true / len(al_words), '.5f'))
print('Доля неправильных переводов и неправильных сопоставлений: ', format(false_false / len(al_words), '.5f'))
print('Доля странных сопоставлений и переводов:     ', format(strange / len(al_words), '.5f'))

f4.write('\n\nResults of  only target sen + mBart + awesome-align + assoc:\n')
f4.write('Доля правильных переводов и правильных сопоставлений:   '+ str( format(true_true / len(al_words), '.5f')) + '\n')
f4.write('Доля правильных переводов и неправильных сопоставлений:   ' + str( format(true_false / len(al_words), '.5f')) + '\n')
f4.write('Доля неправильных переводов и правильных сопоставлений: ' + str( format(false_true / len(al_words), '.5f')) + '\n')
f4.write('Доля неправильных переводов и неправильных сопоставлений: ' + str( format(false_false / len(al_words), '.5f')) + '\n')
f4.write('Доля странных сопоставлений и переводов:     '+  str(format(strange / len(al_words), '.5f')) + '\n')


al_words = unific(al_words, trans)
count_trans = countTranslates2(al_words)

"""Посмотрим, в какой части предложений многозначное слово правильно перевелось и сопоставилось со своим значением, с неправильным значением (то есть переводом другого значения данного слова), а также каков процент непонятных левых слов"""

true_true = 0
true_false = 0
false_true = 0
false_false = 0
strange = 0
t = 0
for i in range(len(al_words)):
  if al_words[i] in translates[all_info.word[i]][str(all_info.gold_sense_id[i])]:
     true_true += 1
  else:
    t = 0
    for el in translates[all_info.word[i]]:
      if al_words[i] in translates[all_info.word[i]][el]:
        false_true += 1
        t = 1
        break
    if (t == 0):
          str1, str2 = re.split('[ ][|]{3}[ ]', trans[i])
          l2 = re.split('[\s]', str2)
          wor = all_info.word[i]
          end = 0
          for w in l2:    
            lemmatizer = WordNetLemmatizer()
            w2 = lemmatizer.lemmatize(w)
            w2 = re.sub(r'[^-a-zA-Z]', r'', w2)
            w2 = w2.lower()
            if w2 in translates[all_info.word[i]][str(all_info.gold_sense_id[i])]:
              true_false += 1
              end = 1
              break
            for j in translates[wor]:
              if w2 in translates[wor][j]:
                false_false += 1
                end = 1
                break
            if end == 1:
              break
          if end == 0:
            strange += 1     
      #print('strange', all_info.word[i], al_words[i])
print('\n\nResults of  only target sen + mBart + awesome-align + assoc + unific:\n')
print('Доля правильных переводов и правильных сопоставлений:   ', format(true_true / len(al_words), '.5f'))
print('Доля правильных переводов и неправильных сопоставлений:   ', format(true_false / len(al_words), '.5f'))
print('Доля неправильных переводов и правильных сопоставлений: ', format(false_true / len(al_words), '.5f'))
print('Доля неправильных переводов и неправильных сопоставлений: ', format(false_false / len(al_words), '.5f'))
print('Доля странных сопоставлений и переводов:     ', format(strange / len(al_words), '.5f'))

f4.write('\n\nResults of  only target sen + mBart + awesome-align + assoc + unific:\n')
f4.write('Доля правильных переводов и правильных сопоставлений:   '+ str( format(true_true / len(al_words), '.5f')) + '\n')
f4.write('Доля правильных переводов и неправильных сопоставлений:   ' + str( format(true_false / len(al_words), '.5f')) + '\n')
f4.write('Доля неправильных переводов и правильных сопоставлений: ' + str( format(false_true / len(al_words), '.5f')) + '\n')
f4.write('Доля неправильных переводов и неправильных сопоставлений: ' + str( format(false_false / len(al_words), '.5f')) + '\n')
f4.write('Доля странных сопоставлений и переводов:     '+  str(format(strange / len(al_words), '.5f')) + '\n')


f4.close()
