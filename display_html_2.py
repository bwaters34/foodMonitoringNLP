#https://seaborn.pydata.org/generated/seaborn.distplot.html
import re 
import front_end 
import pickle 
import sys 
from collections import defaultdict 
import matplotlib.pyplot as plt
import seaborn as sns
from nltk import pos_tag, word_tokenize
from os import path 
def load(fileName):
	with open(fileName, 'r') as f:
		return pickle.load(f) 

def save(variable, fileName):
	with open(fileName, 'w') as f:
		pickle.dump(variable, f)

def read_file(fileName):
	write2file = ''
	foodNames = load(path.join('.', path.join('data','food_pair_dict.pickle')))
	#foodNames = load('.\\data\\nltk_food_dictionary.pickle')
	unique_food_names = {}
	f = file(fileName, 'r')
	for i in f:
		if i[0] == '*':
			text = ''
			i = i.lower()
			#i = i.split()
			#for word in i:
			#	if word not in foodNames:
			#		text += word + ' '
			#	else:
			#		text += '<mark>'+word+'</mark> '
			#write2file += text + '<br>'
			found_at_least = 0
			index_of_food_names = []
			temp_i = re.sub('[^a-zA-Z0-9 \n]', ' ', i[3:])

			for word in foodNames:
				#if word in i:
				if word == 'brownies':
					print word
					#match_word(word, temp_i, 1)
				# if match_word(word, temp_i):

				if temp_i.__contains__(' ' + word + ' '):
					# tags = pos_tag(word_tokenize(temp_i))
					# print(tags)
					print word
					unique_food_names[word] = 1
					found_at_least = 1
					c =  i.find(word) 
					index_of_food_names.append([c ,c + len(word) + 1])
					#print "word found", word, len(word), max_len, max_len_word
			if found_at_least:	
				dic = minimum_no_meeting_rooms(index_of_food_names, len(i))
				for char_pos in dic:
					if dic[char_pos] == 1:
						text += '<mark>' +  i[char_pos] + '</mark>'
					else:
						text += i[char_pos]	
			else:
				pass
				text += i[1:] 
			write2file += text + '<br>'
	#return write2file, unique_food_names
	return write2file

def match_word(food_key_word, sentence, value = 0):
	food_key_word = food_key_word.split()
	sentence = sentence.split()
	for word in food_key_word:
		if word not in sentence:
			return 0
	return 1 

def minimum_no_meeting_rooms(list_of_timings, length_of_sent):
	dic = defaultdict(int)
	for i in xrange(length_of_sent):
		dic[i] = 0
	for meeting_schedules in list_of_timings:
		for i in xrange(meeting_schedules[0], meeting_schedules[1]):
			dic[i] = 1 
	return dic 

def check_if_noun():
	pass


if __name__ == '__main__':
	try:
		# print 4/0
		fileName = 'HSLLD/HV1/MT/admmt1.cha'
		html_format = read_file(fileName)
		#print "HTNL Format", html_format
		front_end.wrapStringInHTMLWindows(body = html_format)
	except:
		print "none"
		print sys.exc_info()
	
	# fileCounts = []
	# all_files = load("C:\\Users\\priti\\OneDrive\\Documents\\CCPP\\FoodMonitoring-NLP\\data\\food_files.pickle")
	# c = 0
	# for file_name in all_files:
	# 	print "File ", c 
	# 	c += 1 
	# 	try:
	# 		html_format, count = read_file(file_name)
	# 	except:
	# 		continue
	# 	else:
	# 		fileCounts.append(len(count))
	# sns.distplot(fileCounts, 
	# 				#hist = False,
	# 				kde = False, 
	# 				#rug=False, 
	# 				norm_hist = False, 
	# 				rug_kws={"color": "g"}, 
	# 				kde_kws={"color": "k", "lw": 3, "label": "KDE"},
	# 				hist_kws={"histtype": "step", "linewidth": 3,"alpha": 1, "color": "g"})
	# plt.show()

