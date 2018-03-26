from pprint import pprint
from nltk import pos_tag

class parse:
	def __init__(self, pattern = 'nltk'):
		if pattern == 'ark':
			self.pattern = './patterns_ark'
		else:
			self.pattern = './patterns'
		self.pattern = self.read_file()
	
	def read_file(self):
		pattern = self.pattern
		pattern = open(pattern, 'r').read().strip().split('\n')
		pattern = [x.strip().split('+') for x in pattern]
		# pprint(pattern)
		return pattern

	def pattern_matching(self, sentence, sentence_pos_tag):
		#print sentence_pos_tag
		#print sentence_pos_tag[1:3]
		return_array = []
		for index_i, i in enumerate(self.pattern):
			# print i
			for index_j, j in enumerate(sentence_pos_tag):
				if index_j+len(i) < len(sentence_pos_tag):
				# if index_j+len(i) < 3:
					temp = sentence_pos_tag[index_j: index_j+len(i)]
					temp_tags = [x[1] for x in temp]
					if i == temp_tags:
						#print "yes"
						# print temp, temp_tags, i
						temp_sentence = ' '.join([x[0] for x in temp]).lower()
						index_sentence = sentence.find(temp_sentence)
						#print temp_tags, temp_sentence, index_sentence, index_sentence + len(temp_sentence), sentence[index_sentence:index_sentence+len(temp_sentence)]
						return_array.append([temp_tags, temp_sentence, index_sentence,index_sentence+len(temp_sentence)])
		return return_array

	def generate_max_two_words(self, sentence, sentence_pos_tag):
		# [[['NNP', 'IN'], 'university of', -1, 12]]
		# print sentence_pos_tag, "\n", sentence
		modified_sentence = ''.join([x if x.isalpha() or x==' ' else '' for x in sentence])
		modified_sentence = modified_sentence.split()
		return_array = []
		# print modified_sentence
		for max_no_words in xrange(1, 3):
			for i, word in enumerate(modified_sentence):
				if i+max_no_words<=len(modified_sentence):
					# print modified_sentence[max_no_words][i+max_no_words]
					# print modified_sentence[i:i+max_no_words]
					end_word = modified_sentence[i:i+max_no_words]
					initial_index = sentence.find(modified_sentence[i])
					if len(end_word) > 1:
						end_word = str(end_word[-1])
						
						final_index = sentence.find(end_word) + len(end_word)
						# print "End word->", end_word, final_index,sentence[initial_index: initial_index+final_index+1]
					else:
						
						end_word = str(end_word[0])
						final_index = sentence.find(end_word) + len(end_word)
						# print "End word->", end_word, final_index,sentence[initial_index: initial_index+final_index+1]
					# print "output->\"", sentence[initial_index: final_index], "\"ends", initial_index, final_index
					return_array.append([sentence_pos_tag[i:i+max_no_words],sentence[initial_index:final_index], initial_index, final_index])
		# print return_array
		return return_array

		# for index_i, i in enumerate():
		# 	pass


if __name__ == '__main__':
	par = parse(pattern='ntlk')
	# text = "I love University of Massachusetts, Amherst in."
	text = "The red dog went to the store"
	pos  = pos_tag(text.split())
	print "Final Array: ", par.pattern_matching(text, pos)
	pprint(par.pattern_matching(text, pos))

