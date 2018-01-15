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
import solution_parser


def load(fileName):
	with open(fileName, 'r') as f:
		return pickle.load(f) 

def save(variable, fileName):
	with open(fileName, 'w') as f:
		pickle.dump(variable, f)

def read_file(fileName):
	write2file = ''
	#Previous versions
	#foodNames = load(path.join('.', path.join('data','food_pair_dict.pickle')))
	#foodNames = load('.\\data\\nltk_food_dictionary.pickle')
	foodNames = load("./data/food_desc_files/food_names.pickle")

	foodGroup = load("./data/food_desc_files/food_group.pickle")

	langua = load("./data/food_desc_files/langua.pickle")

	unique_food_names = {}
	f = file(fileName, 'r')
	current_line_number = 0
	predicted_food_labels_set = set() # syntax: key = (line_number, (start_index_of_food_string_on_line, end_index_of_food_string_on_line), where ending indices are inclusive.
	for line_no, i in enumerate(f): # i is the current line (a string)
		
		food_id_group_pairs = []
		food_id_langua_pairs = []
		current_line_number += 1
		if i[0] == '*':
			word_char_index, word_char_index_string_fromat = provide_words_with_char_nos(i, line_no)
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
			temp_i = re.sub('[^a-zA-Z0-9 \n]', ' ', i[4:])
			#temp_i = i[4:]
			for word in foodNames:
				#if word in i:
				if word == 'brownies':
					print word
					#match_word(word, temp_i, 1)
				# if match_word(word, temp_i):

				if temp_i.__contains__(' ' + word + ' '):
					# print(tags)
					print word
					unique_food_names[word] = 1
					found_at_least = 1
					c =  i.find(word)
					tags = pos_tag(word_tokenize(temp_i))
					individual_food_words = word.split()
					last_word = individual_food_words[-1]
					# for word, label in tags:
					# 	if word == last_word and check_if_noun(label):
					# 		index_of_food_names.append([c, c + len(word) + 1])
					# 		print('chose word: '+ word)
					# 		pass
					# 	else:
					# 		continue
					print(tags)
					print(individual_food_words)
					index_of_food_names.append([c, c + len(word) + 1])
					set_elem = (current_line_number, (c, c + len(word))) # removed the plus one
					predicted_food_labels_set.add(set_elem)

					#Adding stuffs after reading documentation from USDA
					#print ("food -> ", foodNames[word], foodGroup[foodNames[word]])
					food_id = foodNames[word]
					if food_id in foodGroup:
						food_group_for_food_id = foodGroup[food_id]
						food_id_group_pairs.append([word, food_group_for_food_id])
					
					if food_id in langua:
						temp_langua = langua[food_id]
						t = []
						for temp_words in temp_langua:
							t.append(temp_words)
						food_id_langua_pairs.append([word + " " + food_id, t])
						#food_id_langua_pairs = 
					print("food -> ", food_id_group_pairs)
			#print "word found", word, len(word), max_len, max_len_word
			#print ("Temproray -> ", temp_i)
			#print ("Final i -> ", i)
			if found_at_least:	
				dic = minimum_no_meeting_rooms(index_of_food_names, len(i))
				print('dic')
				print(dic)
				for char_pos in dic:
					if dic[char_pos] == 1:
						text += '<mark>' +  i[char_pos] + '</mark>'
					else:
						text += i[char_pos]
			else:
				pass
				text += i[1:] 
			#print ("Final text ->", text)
			tags = pos_tag(word_tokenize(temp_i))
			#Joining the tags
			tags = join_tags(tags)
			#print("tags -> ", tags)
			#print("pairs ---> ", food_id_langua_pairs, len(food_id_langua_pairs))

			print ("pairs -> ", word_char_index)

			food_tags = ''
			if len(food_id_group_pairs):
				for pairs in food_id_group_pairs:
					food_tags += "<mark>" + pairs[0] + "</mark>" + "----> " + pairs[1] + "<br>"
			food_ledger_langua = ''
			if len(food_id_langua_pairs):
				for pairs in food_id_langua_pairs:
					food_name_langua = pairs[0]
					food_ledger_langua += "<mark>" + food_name_langua + "----></mark>"
					for ledger in pairs[1]:
						food_ledger_langua += ledger.lower() + ",  "
					food_ledger_langua += "<br>" + "<br>"
			write2file += text + word_char_index_string_fromat + '<br>' + tags + '<br>' + food_tags + '<br>' + food_ledger_langua 

			#Orignal 
			#write2file += text + '<br>'

	# attempt to load solutions file
	solution_file_path = path.join('solutions', fileName)
	try:
		print('loading solution set')
		solution_set = solution_parser.get_solution_set_from_file(solution_file_path)
		print('calculating')
		precision, recall, false_pos_list, false_neg_list, true_pos_list = solution_parser.calculate_precision_and_recall(solution_set, predicted_food_labels_set)
		print('precision: ' +  str(precision))
		print('recall: ' +  str(recall))
		print('true positives:') + str(true_pos_list)
		print('lines:')
		for line in solution_parser.get_corresponding_lines(fileName, true_pos_list):
			print(line)
		print('false positives: ' + str(false_pos_list))
		print('lines:')
		for line in solution_parser.get_corresponding_lines(fileName, false_pos_list):
			print(line)
		print('false negatives: ' + str(false_neg_list))
		print('lines:')
		for line in solution_parser.get_corresponding_lines(fileName, false_neg_list):
			print(line)


	except IOError:
		print('no solution file found for: ' + solution_file_path)
	#return write2file, unique_food_names
	return write2file

def provide_words_with_char_nos(sentence, line_no):
	temp_char = ''
	start_count = 0 
	return_array = []
	for index, char in enumerate(sentence):
		if char != ' ' and char != '\t':
			temp_char += char 
		else:
			return_array.append([temp_char, start_count, index])
			start_count = index + 1 
			temp_char = ' '

	#Converting to displayable format (String format)
	return_string = '<br>(line->' + str(line_no)+ ") "
	for word in return_array:
		return_string += word[0].lower() + " ("+str(word[1])+","+str(word[2])+") "
	return_string += "<br>"
	return return_array, return_string



def join_tags(sentence):
	text = '     '
	for i in sentence:
		text += '(' + i[0] + "->" + i[1] + ") "
	return text

def match_word(food_key_word, sentence, value = 0):
	food_key_word = food_key_word.split()
	sentence = sentence.split()
	for word in food_key_word:
		if word not in sentence:
			return 0
	return 1 

def minimum_no_meeting_rooms(list_of_timings, length_of_sent):
	dic = defaultdict(int)
	for i in xrange(1, length_of_sent):
		dic[i] = 0
	for meeting_schedules in list_of_timings:
		for i in xrange(meeting_schedules[0], meeting_schedules[1]):
			dic[i] = 1 
	return dic 

def check_if_noun(tag):
	if tag == 'NN' or tag == 'NNS' or tag == 'NNP' or tag == 'NNPS':
		return True
 	return False



if __name__ == '__main__':
	try:
		#fileName = 'HSLLD/HV3/MT/brtmt3.cha' # coffee
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

