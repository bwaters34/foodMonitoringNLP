from collections import defaultdict
import numpy as np

class levenshtein_distance:
	def __init__(self, distance_cache = None, **weight_dict):
		alphabet = "abcdefghijklmnopqrstuvwxyz "
		self.w = defaultdict(lambda:(1,1,1))
		# self.w = dict( (x, (1, 1, 1)) for x in alphabet + alphabet.upper())
		if weight_dict:
			self.w.update(weight_dict)
		if distance_cache is None:
			self.distance_cache = {}
		else:
			self.distance_cache = distance_cache
		

	def calculate_distance(self, s, t):
		# return 0
		# return dist[row][col]
		rows = len(s)+1
		cols = len(t)+1
		# print cols 
		dist = np.zeros((rows, cols), dtype=int)

		# see if any part of s and t are in the distance cache
		stored = None # we didn't change anything
		print('cache:')
		print(self.distance_cache)
		for i in xrange(1,min(len(t),len(s))+1):
			s_substring = s[:i]
			t_substring = t[:i]
			print('searching cache')
			print(s_substring, t_substring)
			if (s_substring, t_substring) in self.distance_cache:
				stored = self.distance_cache[(s_substring, t_substring)]
			else:
				print('could not find:')
				print(s_substring, t_substring)
				print(self.distance_cache)
				break

		if stored is None:
			num_stored_rows = 0
			num_stored_cols = 0
		else:
			num_stored_rows = len(stored)
			num_stored_cols = len(stored[0])
			# np.copyto(dst=dist, src=stored)
			dist[0:stored.shape[0], 0:stored.shape[1]] = stored # copy stored into dist

		for row in range(1, rows):
			dist[row][0] = dist[row-1][0] + self.w[s[row-1]][0]
		for col in range(1, cols):
			dist[0][col] = dist[0][col-1] + self.w[t[col-1]][1]
		# self.display(dist)
		# print rows,colss
		print('num stored cols, num stored rows')
		print(num_stored_cols, num_stored_rows)
		for row in range(1, rows): # TODO: we need to evaluate the far right column, but also the bottom row.
			for col in xrange(1,cols): # TODO: does this even save us any time???
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
		self.display(dist)
		# cache new stuff
		# just gonna cache the squares only, because I'm lazy and it saves a lot of memory
		for i in xrange(num_stored_rows, min(len(dist), len(dist[0]))):
			s_substring = s[:i]
			t_substring = t[:i]
			self.distance_cache[(s_substring, t_substring)] = dist[:i+1, :i+1]

		return dist[rows-1][cols-1]

	def display(self, array):
		for i in array:
			print i 

def get_levenshtein_distance_object(setting = 'system1'):
	if setting == 'system1':
		ld = levenshtein_distance(a=(3, 3, 1),
                            e=(3, 3, 1),
                            i=(3, 3, 1),
                            o=(3, 3, 1),
                            u=(3, 3, 1),
                            s=(0, 0, 1))
		return ld
	elif setting == 'system2':
		ld = levenshtein_distance()
		return ld
	elif setting == 'system3':
		ld = levenshtein_distance(s = (0, 0, 1))
		return ld
	else:
		print(setting)
		raise ValueError

if __name__ == '__main__':
	# ld = levenshtein_distance(a=(3, 3, 1),
 #                            e=(3, 3, 1),
 #                            i=(3, 3, 1),
 #                            o=(3, 3, 1),
 #                            u=(3, 3, 1),
 #                            s=(0, 0, 1))
 # 	ld = levenshtein_distance(s = (0, 0, 1))
	# print ld.calculate_distance("abx", "xya")
	arr = np.array([[1,2],[3,4]])
	print(arr)
	print(arr[0:1,0:2])
	a = get_levenshtein_distance_object(setting = 'system2')
	print(a.calculate_distance('gumbo', 'gambol'))
	print(a.calculate_distance('gumbo', 'gambol'))
	# print(a.calculate_distance('gumbo', 'gambol'))
	# print(a.calculate_distance('gumbo', 'gambol'))
