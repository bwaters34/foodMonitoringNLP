import seaborn as sns
from collections import Counter
import cPickle as pickle
results = pickle.load(open('most_recent_results.pickle', 'rb'))
fn = results.false_neg_list

filtered = list(filter(lambda x: len(x) > 0, fn)) # just lists that contain more than one item
flat_list = [item for sublist in filtered for item in sublist] # just one list of strings (contains duplicates!)
size_of_ngram = [len(word.split()) for word in flat_list]
cnt = Counter()
for num in size_of_ngram:
    cnt[num] +=1
print(cnt)

results = list(cnt.iteritems())
x, y = ([ a for a,b in results], [ b for a,b in results ])
print(x)
print(y)
y_percent = [elem / float(sum(y)) for elem in y]
ax = sns.barplot(x,y_percent, )
ax.set(xlabel='ngram length', ylabel='frequency (%)')
ax.set(title='Size of ngrams in False Negatives')
print(y_percent)
# unique_only = list(set(flat_list))
# print(len(unique_only))
# print(len(flat_list))
# size_of_ngram = [len(word.split()) for word in unique_only]
# cnt = Counter()
# for num in size_of_ngram:
#     cnt[num] +=1
# print(cnt)
#
# results = list(cnt.iteritems())
# x, y = ([ a for a,b in results], [ b for a,b in results ])
# print(x)
# print(y)
# ax = sns.barplot(x,y, )
# ax.set(xlabel='ngram length', ylabel='frequency')
# ax.set(title='Size of ngrams in False Negatives (unique words)')
