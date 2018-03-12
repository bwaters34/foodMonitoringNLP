def match_strings(word1, word2):
	temp = ''.join([x if x.isalpha() or x == ' ' else '' for x in word1]).strip()
	temp2 = ''.join([x if x.isalpha() or x== ' ' else '' for x in word2]).strip()
	# print temp[-1], temp[:-1]
	print temp, temp2
	if temp[-1] == 's' or temp2[-1] == 's':
		print "goes", temp[:-1], temp2
		print "goes", temp, temp2[:-1]
		if temp[:-1] == temp2:
			print "yes if 1", temp[:-1], temp2
		elif temp == temp2[:-1]:
			print "yes elif 2", temp, temp2[-1:]
		else:
			print "No but from 's' factor"
	elif temp == temp2:
		print "yes simple match", temp, temp2
	else:
		print "No"

	if temp[-2:] == 'es' or temp2[-2:] == 'es':
		if temp[:-2] == temp2:
			print "yes if 1"
		elif temp == temp2[-2:]:
			print "yes elif 2", temp, temp2[-2:]
		else:
			print "No but from 's' factor"
	elif temp == temp2:
		print "yes simple match", temp, temp2
	else:
		print "No"
	
	print "Output is ->",temp, temp2
#Main 
# match_strings("<applesauces>", "applesauce")
# match_strings("potatoes", "potato")
# match_strings("(to)matoes", "tomato")
# match_strings("cucum(ber)", "cucumber")
match_strings("milk shake", "<milk sha<kes>")