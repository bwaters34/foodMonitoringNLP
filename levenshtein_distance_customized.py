class levenshtein_distance:
	def __init__(self, **weight_dict):
		alphabet = "abcdefghijklmnopqrstuvwxyz "
		self.w = dict( (x, (1, 1, 1)) for x in alphabet + alphabet.upper())
		if weight_dict:
			self.w.update(weight_dict) 
		# print self.w 
		

	def calculate_distance(self, s, t):
		# return 0
		# return dist[row][col]
		rows = len(s)+1
		cols = len(t)+1
		# print cols 
		dist = [[0 for x in range(cols)] for x in range(rows)]
		for row in range(1, rows):
			dist[row][0] = dist[row-1][0] + self.w[s[row-1]][0]
		for col in range(1, cols):
			dist[0][col] = dist[0][col-1] + self.w[t[col-1]][1]
		# self.display(dist)
		# print rows,colss
		for col in range(1, cols):
			for row in xrange(1, rows):
				deletes = self.w[s[row - 1]][0]
				inserts = self.w[t[col - 1]][1]
				subs = max( self.w[s[row-1]][2], self.w[t[col-1]][2])
				if s[row-1] == t[col-1]:
					subs = 0 
				else:
					subs = subs
				dist[row][col] = min(dist[row-1][col] + deletes,
									dist[row][col-1] + inserts,
									dist[row-1][col-1] + subs)
		# self.display(dist)
		return dist[rows-1][cols-1]

	def display(self, array):
		for i in array:
			print i 

if __name__ == '__main__':
	pass	
	# ld = levenshtein_distance(a=(3, 3, 1),
 #                            e=(3, 3, 1),
 #                            i=(3, 3, 1),
 #                            o=(3, 3, 1),
 #                            u=(3, 3, 1),
 #                            s=(0, 0, 1))
 	ld = levenshtein_distance(s = (0, 0, 1))
	print ld.calculate_distance("abx", "xya")