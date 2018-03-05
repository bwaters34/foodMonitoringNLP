def display(array):
	for i in array:
		print i
def levenshtien_distance(s, l, delete = 1, insertion = 1, substitution = 1):
	array = [[0 for x in xrange(len(l) + 1)] for x in xrange(len(s) + 1)]
	for i in xrange(len(l) + 1):
		array[i][0] = i
	for i in xrange(len(s) + 1):
		array[0][i] = i 
	# display(array)

	for row in xrange(1, len(s) + 1):
		for col in xrange(1, len(l) + 1):
			if s[row - 1] == l[col - 1]:
				cost = 0
			else:
				cost = substitution
			temp = min(array[row - 1][col] + delete, #Deletion, 
									array[row][col - 1] + insertion, #Insertion
									array[row - 1][col - 1] + cost) #Subsitute)
			array[row][col] = temp
		# print s[row-1], l[col - 1], cost,  row, col, temp
	display(array)
		# print "\n\n"

	
if __name__ == '__main__':
	# levenshtien_distance("flaw", "lawn")
	levenshtien_distance("abc", "xyz", 1, 1, 1)