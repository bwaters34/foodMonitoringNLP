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
from collections import namedtuple
import solution_parser
import CMUTweetTagger
import os
import cal_calorie_given_food_name 
import parse 
import nltk 
import time 

def load(fileName):
	with open(fileName, 'r') as f:
		return pickle.load(f) 

def save(variable, fileName):
	with open(fileName, 'w') as f:
		pickle.dump(variable, f)

def read_file(fileName, parser_type = None, only_files_with_solutions = False, base_accuracy_on_how_many_unique_food_items_detected = True, use_second_column = True):
	write2file = ''
	total_calorie = 0.0
	calorie = cal_calorie_given_food_name.food_to_calorie()
	par = parse.parse()

	#Previous versions
	#foodNames = load(path.join('.', path.join('data','food_pair_dict.pickle')))
	#foodNames = load('.\\data\\nltk_food_dictionary.pickle')
	foodNames = load("./data/food_desc_files/food_names.pickle")
	extraFoodNames = load("./data/food_desc_files/extra_food_names.pickle")
	print('adding extra names')
	print(len(foodNames))
	print(len(extraFoodNames))
	foodNames.update(extraFoodNames)
	print(len(foodNames))
	foodGroup = load("./data/food_desc_files/food_group.pickle")
	langua = load("./data/food_desc_files/langua.pickle")

	ark_parsed_data = ark_parser(fileName)

	unique_food_names = {}
	f = file(fileName, 'r')
	current_line_number = 0
	predicted_food_labels_set = set() # syntax: key = (line_number, (start_index_of_food_string_on_line, end_index_of_food_string_on_line), where ending indices are inclusive.
	solution_set_loaded = False
	solution_file_path = path.join('solutions', fileName)
	try:
		print('loading solution set')
		solution_set = solution_parser.get_solution_set_from_file(solution_file_path)
		solution_set_loaded = True
	except IOError:
		print('no solution file found for: ' + solution_file_path)

	# if we only want files with solutions, and no solution set is found, break early so we don't need to parse the file for food words.
	if only_files_with_solutions:
		if not solution_set_loaded:
			return "solution set not found", None
	for line_no, i in enumerate(f): # i is the current line (a string)
		calorie_text = ''
		food_id_group_pairs = []
		food_id_langua_pairs = []
		current_line_number += 1
		if i[0] == '*':
			print "LINE NO -> ", line_no
			word_char_index, word_char_index_string_fromat = provide_words_with_char_nos(i, line_no+1)
			text = ''
			edit_distance_i = i
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
			spans_found_on_line = []

			#FOR EDIT DISTANCE
			sentence_pos_tags = par.pattern_matching(edit_distance_i, pos_tag(edit_distance_i.split()))
			#print "ATTENTION", sentence_pos_tags
			for word in foodNames:
				if temp_i.__contains__(' ' + word + ' '):
					# print(tags)
					print word
					unique_food_names[word] = 1
					found_at_least = 1
					
					# #Previous Setting
					# c =  i.find(word)
					# index_of_food_names.append([c, c + len(word) + 1])
					
					# #removed the plus one
					# spans_found_on_line.append((c, c + len(word)))
					try:
						temp_calorie = calorie.cal_calorie(word)
						total_calorie += temp_calorie
						calorie_text += '<br><mark>'+word+"</mark>-> "+str(temp_calorie)
					except:
						print sys.exc_info()
						pass 

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


					for match in re.finditer(word, i):
						food_match_indexes = match.span()
						index_of_food_names.append([food_match_indexes[0], food_match_indexes[1]])
						spans_found_on_line.append([food_match_indexes[0], food_match_indexes[1]])
						
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
				#Checking for EDIT Distance
				if len(sentence_pos_tags) > 0:
					# if word == 'carrot':
					# 	print "CARROT"
					# if word == 'tomatoes':
					# 	print "DIAGONISING", sentence_pos_tags, word
					for food_data in sentence_pos_tags:
						k1 = float(len(food_data[1]))/float(len(word))
						if 0.6 < k1 and k1 < 1.4:
						#if 0.6 < k and k < 1.4:
						#k1 = abs(len(food_data[1]) - len(word)) 
						#if k1 <= 3:
							# if word == 'tomatoes':
							# 	print word, food_data[1], "Reached first pass",  nltk.edit_distance(word, food_data[1])
							#print "yes", food_data[1], word
							#PERFORM EDIT DISTANCE
							distance = nltk.edit_distance(word, food_data[1])
							k2 = distance/float(max(len(word), len(food_data))) 
							if k2 <=0.3:
							#k2 = 3
							#if distance <= k2:
								found_at_least = 1
								# if word == 'tomatoes':
								# 	print word, food_data[1], "Reached SECOND pass",  nltk.edit_distance(word, food_data[1])
								index_of_food_names.append([food_data[2], food_data[3]])
								spans_found_on_line.append([food_data[2], food_data[3]])
								#with open("./notes/edit_distance_3.txt", "a") as myfile:
								with open("./notes/edit_distance_30_percen.txt", "a") as myfile:
									myfile.write(word +"," + food_data[1] + ","+ str(distance) + ", "+ str(k1) + " , "+ str(k2) + "\n")
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
				text += calorie_text

				tuples_list = give_largest_non_overlapping_sequences(spans_found_on_line)  # filters out spans that conflict with other spans. larger spans are given priority
				for tup in tuples_list:
					set_elem = (current_line_number, tup) # add line number so we know where in the document we got it
					predicted_food_labels_set.add(set_elem)
			else:
				pass
				text += i[1:] 
			#print ("Final text ->", text)
			tags = ''
			if parser_type == 'stanford_POS' and 0:
				# print('running stanford')
				tags = pos_tag(word_tokenize(temp_i))
				#Joining the tags
				tags = join_tags(tags)
			elif parser_type == 'ark_tweet_parser' and 0:
				print('running ark')
				#tags =  CMUTweetTagger.runtagger_parse([temp_i])
				tags = join_tags(ark_parsed_data[line_no])
				#tags = ''
				#tags1 = join_tags(tags)

			#print("tags -> ", tags1)
			#print("pairs ---> ", food_id_langua_pairs, len(food_id_langua_pairs))

			#print ("pairs -> ", word_char_index)

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
	write2file += "<hr>" + "Total Calories -> " + str(total_calorie) 
	num_true_pos = None # give dummy values in case try fails
	num_false_pos = None
	num_false_neg = None
	if solution_set_loaded:
		print('loading solution set')
		solution_set = solution_parser.get_solution_set_from_file(solution_file_path)
		print('calculating')
		if base_accuracy_on_how_many_unique_food_items_detected:
			food_names_only_solution_set = solution_parser.convert_solution_set_to_set_of_food_names(fileName, solution_set)
			food_names_only_predicted_set = solution_parser.convert_solution_set_to_set_of_food_names(fileName, predicted_food_labels_set)
			precision, recall, false_pos_list, false_neg_list, true_pos_list = solution_parser.calculate_precision_and_recall(
				food_names_only_solution_set, food_names_only_predicted_set)
		else:
			precision, recall, false_pos_list, false_neg_list, true_pos_list = solution_parser.calculate_precision_and_recall(solution_set, predicted_food_labels_set)
		num_true_pos = len(true_pos_list)
		num_false_pos = len(false_pos_list)
		num_false_neg = len(false_neg_list)
		print('file:' + fileName)
		print('precision: ' +  str(precision))
		print('recall: ' +  str(recall))
		print('true positives:') + str(true_pos_list)
		if not base_accuracy_on_how_many_unique_food_items_detected:
			for line in solution_parser.get_corresponding_lines(fileName, true_pos_list):
				print(line)
		print('false positives: ' + str(false_pos_list))
		if not base_accuracy_on_how_many_unique_food_items_detected:
			for line in solution_parser.get_corresponding_lines(fileName, false_pos_list):
				print(line)
		print('false negatives: ' + str(false_neg_list))
		if not base_accuracy_on_how_many_unique_food_items_detected:
			for line in solution_parser.get_corresponding_lines(fileName, false_neg_list):
				print(line)
		print('# true pos: {}'.format(num_true_pos))
		print('# false pos: {}'.format(num_false_pos))
		print('# false neg: {}'.format(num_false_neg))

		if not base_accuracy_on_how_many_unique_food_items_detected:
			write2file += '<br><hr>'+"Precision: "+str(precision)+ \
							"<br>Recall: "+str(recall) + "<br><hr>"
			write2file += 	"False Positives<br>"+ str(false_pos_list)+ \
							"<br>"
			for line in solution_parser.get_corresponding_lines(fileName, false_pos_list):
				write2file += str(line)+ " ---> <mark>" + str(line[1][line[0][1][0]:line[0][1][1]]) +"</mark><br>"
			write2file +=	"<hr>False negatives:<br>"+str(false_neg_list) + "<br>"
			for line in solution_parser.get_corresponding_lines(fileName, false_neg_list):
				write2file += str(line)+ " ---> <mark>" + str(line[1][line[0][1][0]:line[0][1][1]]) +"</mark><br>"

	else:
		print('no solution set found')
	#return write2file, unique_food_names
	#namedtuple()

	Accuracy = namedtuple('Accuracy',
						  'num_true_pos num_false_pos num_false_neg')  # makes returning multiple values more clear
	results = Accuracy(num_true_pos=num_true_pos, num_false_pos=num_false_pos, num_false_neg=num_false_neg)


	return write2file, results


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

def give_largest_non_overlapping_sequences(list_of_start_end_tuples):
	Sequence = namedtuple('Sequence', ['start', 'end', 'size'])
	list_of_named_sequences = [Sequence(start = x[0], end = x[1], size = x[1] - x[0] - 1) for x in list_of_start_end_tuples] # size is -1 because the end number represents the index of the character AFTER the last character in the sequence.
	sorted_by_size_sequences = sorted(list_of_named_sequences, key=lambda seq: seq.size) # smallest size is first, largest size is last
	non_overlapping_sequences = []
	while len(sorted_by_size_sequences) > 0:
		sequence = sorted_by_size_sequences.pop() # last element in list, therefore sequence with largest size still on the list
		if not conflicts_with_sequences(non_overlapping_sequences, sequence):
			non_overlapping_sequences.append(sequence)
	extracted_tuples = [(seq.start, seq.end) for seq in non_overlapping_sequences]
	return extracted_tuples

def conflicts_with_sequences(list_of_sequences, test_sequence):
	"""Tests if test_sequence conflicts with any sequence in the list_of_sequences"""
	for already_added_sequence in list_of_sequences:
		if sequences_overlap(already_added_sequence, test_sequence):
			return True
	return False

def sequences_overlap(seq1, seq2):
	"""Returns if two sequences overlap"""
	if seq1.end <= seq2.start: # seq1 must end before seq2 begins. they do not overlap
		return False
	elif seq2.end <= seq1.start:
		return False
	else:
		return True

def ark_parser(fileName):
	final_list_of_sentences = []
	list_of_sentences = open(fileName, "r").read()
	for sentence in list_of_sentences.split('\n'):
		if len(sentence) > 1:
			if sentence[0] == '*':
				final_list_of_sentences.append(' '.join(sentence.split()))
	print final_list_of_sentences
	var = CMUTweetTagger.runtagger_parse(final_list_of_sentences)
	return var

def evaluate_all_files_in_directory(directory_path, only_files_with_solutions = False):
	sum_true_pos = 0
	sum_false_pos = 0
	sum_false_neg = 0

	for filename in os.listdir(directory_path):
		file_path = directory_path + '/' + filename
		print(file_path)
		html_format, results = read_file(file_path, only_files_with_solutions=only_files_with_solutions)
		if results is not None:
			if results.num_true_pos is not None:  # if it is none, a solution set was not loaded
				sum_true_pos += results.num_true_pos
			if results.num_false_pos is not None:
				sum_false_pos += results.num_false_pos
			if results.num_false_neg is not None:
				sum_false_neg += results.num_false_neg

	precision = sum_true_pos / float(sum_true_pos + sum_false_pos)
	recall = sum_true_pos / float(sum_true_pos + sum_false_neg)
	return precision, recall, sum_true_pos, sum_false_pos, sum_false_neg


if __name__ == '__main__':
	try:
		#fileName = 'HSLLD/HV3/MT/brtmt3.cha' # coffee
		start = time.time()
		fileName = 'HSLLD/HV1/MT/conmt1.cha'
		html_format, results = read_file(fileName, 'ark_tweet_parser')
		#print "HTNL Format", html_format
		print "Time taken to run the script", time.time() -start
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
	# 		fileCounts.append(len(cont))
	# sns.distplot(fileCounts, 
	# 				#hist = False,
	# 				kde = False, 
	# 				#rug=False, 
	# 				norm_hist = False, 
	# 				rug_kws={"color": "g"}, 
	# 				kde_kws={"color": "k", "lw": 3, "label": "KDE"},
	# 				hist_kws={"histtype": "step", "linewidth": 3,"alpha": 1, "color": "g"})
	# plt.show()

