import sys

def calculate_precision_and_recall(gold_standard_set, predicted_set):
	"""

	:param gold_standard_set:
	:param predicted_set:
	:return:
	"""
	print('calculating accuracy and recall')
	print(gold_standard_set)
	print(predicted_set)
	true_positives = 0
	false_positives = 0
	false_negatives = 0
	for elem in predicted_set:
		if elem in gold_standard_set:
			true_positives += 1
		else:
			false_positives += 1
	for gold_elem in gold_standard_set:
		if gold_elem not in predicted_set:
			false_negatives += 1
	precision = true_positives / float(true_positives + false_positives)
	recall = true_positives / float(true_positives + false_negatives)
	return precision, recall

def get_solution_set_from_file(file_path):
	"""
	returns a set of (line_number, (food_phrase_start_index, food_phrase_end_index))
	Each line of the solution file should contain 3 numbers, the line number of the food phrase, the starting index of the food phrase, and the ending index of the food phrase.
	files are stored under their original name, but in the solutions directory.
	Example file:
	51 23 28
	57 37 52

	:param file_path: file path of solution
	:return: set of solutions, where a set contains
	"""
	solutions = set()
	with open(file_path) as f:
		for line in f:
			words = [int(word) for word in line.split()]
			line_number = words[0]
			word_start = words[1]
			word_end = words[2]
			key = (line_number, (word_start, word_end))
			solutions.add(key)
	return solutions

def creating_solution_helper(file_path):
	# a visual helper to annotate files easier
	with open(file_path) as f:
		line_number = 0
		for line in f:
			line_number += 1
			print("line number: " + str(line_number))
			print(line)
			output_str = ''
			i = 0
			while i < len(line):
				if i % 5 == 0:
					length_of_addition_to_str = len(str(i))
					output_str += str(i)
					i += length_of_addition_to_str
				else:
					output_str += ' '
					i += 1
			print(output_str)

if __name__ == '__main__':
	file_name = sys.argv[1]
	creating_solution_helper(file_name)
