from pprint import pprint
from nltk import pos_tag

class parse:
	def __init__(self):
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
			for index_j, j in enumerate(sentence_pos_tag):
				if index_j+len(i) < len(sentence_pos_tag):
					temp = sentence_pos_tag[index_j: index_j+len(i)]
					temp_tags = [x[1] for x in temp]
					if i == temp_tags:
						#print "yes"
						#print temp, temp_tags, i
						temp_sentence = ' '.join([x[0] for x in temp]).lower()
						index_sentence = sentence.find(temp_sentence)
						#print temp_tags, temp_sentence, index_sentence, index_sentence + len(temp_sentence), sentence[index_sentence:index_sentence+len(temp_sentence)]
						return_array.append([temp_tags, temp_sentence, index_sentence,index_sentence+len(temp_sentence)])
		return return_array

if __name__ == '__main__':
	par = parse()
	text = "the red dog went to the store."
	pos  = pos_tag(text.split())
	print par.pattern_matching(text, pos)

