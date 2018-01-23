import CMUTweetTagger
import time 
fileName = 'HSLLD/HV1/MT/admmt1.cha'

final_list_of_sentences = []
list_of_sentences = open(fileName, "r").read()
for sentence in list_of_sentences.split('\n'):
	if len(sentence) > 1:
		if sentence[0] == '*':
			#print sentence
			final_list_of_sentences.append(' '.join(sentence.split()))

print final_list_of_sentences
start = time.time()
var = CMUTweetTagger.runtagger_parse(final_list_of_sentences)
print "Time taken ", time.time() - start
print var
