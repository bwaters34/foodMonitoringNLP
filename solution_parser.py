import sys

def calculate_precision_and_recall(gold_standard_set, predicted_set):
	"""

	:param gold_standard_set:
	:param predicted_set:
	:return:
	"""
	print('calculating accuracy and recall')
	print('correct labels:')
	print(gold_standard_set)
	print('predicted labels:')
	print(predicted_set)
	true_positives = 0
	false_positives = 0
	false_negatives = 0
	true_pos_list = []
	false_pos_list = []
	false_neg_list = []
	for elem in predicted_set:
		if elem in gold_standard_set:
			true_positives += 1
			true_pos_list.append(elem)
		else:
			false_positives += 1
			false_pos_list.append(elem)
	for gold_elem in gold_standard_set:
		if gold_elem not in predicted_set:
			false_negatives += 1
			false_neg_list.append(gold_elem)

	if true_positives == 0:
		print("OH NO")
	if true_positives + false_positives != 0:
		precision = true_positives / float(true_positives + false_positives)
	else:
		precision = 1.0
	if true_positives + false_negatives != 0:
		recall = true_positives / float(true_positives + false_negatives)
	else:
		recall = 1.0

	return precision, recall, sorted(false_pos_list), sorted(false_neg_list), sorted(true_pos_list)

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
			if len(line) > 1: # make sure its not end of file
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
			if line[0] == '*':
				print('')
				print(line[:-1])
				print("line number: " + str(line_number))
				output_str = ''
				for i in range(len(line[:-1])):
					char = line[i]
					output_str+= char + str(i) + ' '
				print(output_str)

def extract_line_numbers_from_solution_set(solution_set):
	return [x[0] for x in solution_set]


def get_corresponding_lines(file_name, solution_set):
	lines = []

	sorted_list_of_line_numbers = sorted(extract_line_numbers_from_solution_set(solution_set))
	solution_set = sorted(solution_set)
	with open(file_name) as f:
		current_line_number = 0
		current_sorted_line_index = 0
		for line in f:
			current_line_number +=1
			if current_sorted_line_index < len(sorted_list_of_line_numbers) and current_line_number == sorted_list_of_line_numbers[current_sorted_line_index]:
				while current_sorted_line_index < len(sorted_list_of_line_numbers) and sorted_list_of_line_numbers[current_sorted_line_index] == current_line_number:
					lines.append((solution_set[current_sorted_line_index], line))
					current_sorted_line_index += 1
	return lines

def convert_solution_set_to_set_of_food_names(file_path, solution_set):
	food_names = set()
	print(solution_set)
	tuples_and_lines = get_corresponding_lines(file_path, list(solution_set))
	print(tuples_and_lines)
	for t_and_l in tuples_and_lines:
		solution_tuple, line = t_and_l
		substring_indexes = solution_tuple[1]
		start_index, stop_index = substring_indexes # unpack the values from the tuple
		if stop_index < start_index:
			raise ValueError("stop index cannot be before start index, solutions are incorrect:, " + str(solution_tuple))
		food_name = line[start_index:stop_index].lower()
		food_names.add(food_name)

	return food_names

if __name__ == '__main__':
	file_name = sys.argv[1]
	creating_solution_helper(file_name)

