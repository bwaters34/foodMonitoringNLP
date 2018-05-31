
#!/usr/bin/python
#By Steve Hanov, 2011. Released to the public domain
import time
import sys
from collections import defaultdict

# The Trie data structure keeps a set of words, organized with one node for
# each letter. Each node has a branch for each letter that may follow it in the
# set of words.
class LevenshteinTrie:
	def __init__(self, food_words, distance_cache = None, **weight_dict):
		alphabet = "abcdefghijklmnopqrstuvwxyz "
		self.w = defaultdict(lambda:(1,1,1))
		# self.w = dict( (x, (1, 1, 1)) for x in alphabet + alphabet.upper())
		if weight_dict:
			self.w.update(weight_dict)
		if distance_cache is None:
			self.distance_cache = {}
		else:
			self.distance_cache = distance_cache
		self.trie = TrieNode()
		self.wordcount = 0
		for word in food_words:
			self.wordcount += 1
			self.trie.insert(word)
		self.nodecount = 0

	# The search function returns a list of all words that are less than the given
	# maximum distance from the target word
	def search(self, word, maxCost):

		# build first row
		currentRow = range(len(word) + 1)

		results = []

		# recursively search each branch of the trie
		for letter in self.trie.children:
			self.searchRecursive(self.trie.children[letter], letter, word, currentRow,
							results, maxCost)

		return results

	# This recursive helper is used by the search function above. It assumes that
	# the previousRow has been filled in already.
	def searchRecursive(self, node, letter, word, previousRow, results, maxCost):

		columns = len(word) + 1
		currentRow = [previousRow[0] + 1]

		# Build one row for the letter, with a column for each letter in the target
		# word, plus one for the empty string at column 0
		for column in xrange(1, columns):
			# deletes = self.w[s[row - 1]][0] #
			deletes = self.w[word[column - 1]][0]
			# inserts = self.w[t[col - 1]][1]
			inserts = self.w[letter][1]
			subs = max(self.w[word[column - 1]][2], self.w[letter][2])

			insertCost = currentRow[column - 1] + inserts
			deleteCost = previousRow[column] + deletes

			if word[column - 1] != letter:
				replaceCost = previousRow[column - 1] + subs
			else:
				replaceCost = previousRow[column - 1]

			currentRow.append(min(insertCost, deleteCost, replaceCost))

		# if the last entry in the row indicates the optimal cost is less than the
		# maximum cost, and there is a word in this trie node, then add it.
		if currentRow[-1] <= maxCost and node.word != None:
			results.append((node.word, currentRow[-1]))

		# if any entries in the row are less than the maximum cost, then
		# recursively search each branch of the trie
		if min(currentRow) <= maxCost:
			for letter in node.children:
				self.searchRecursive(node.children[letter], letter, word, currentRow,
								results, maxCost)


class TrieNode:
    def __init__(self):
        self.word = None
        self.children = {}

    def insert(self, word ):
        node = self
        for letter in word:
            if letter not in node.children:
                node.children[letter] = TrieNode()

            node = node.children[letter]

        node.word = word

def get_levenshtein_distance_object(food_words, setting = 'system1', ):
	if setting == 'system1':
		ld = LevenshteinTrie(food_words, a=(3, 3, 1),
                            e=(3, 3, 1),
                            i=(3, 3, 1),
                            o=(3, 3, 1),
                            u=(3, 3, 1),
                            s=(0, 0, 1))
		return ld
	elif setting == 'system2':
		ld = LevenshteinTrie(food_words)
		return ld
	elif setting == 'system3':
		ld = LevenshteinTrie(food_words, s = (0, 0, 1))
		return ld
	else:
		print(setting)
		raise ValueError
if __name__ == '__main__':

	start = time.time()
	lt = get_levenshtein_distance_object(food_words='cat dog gumbo gambol'.split(), setting='system2')
	results = lt.search('gumbo', 5000)
	end = time.time()

	for result in results: print result

	print "Search took %g s" % (end - start)

