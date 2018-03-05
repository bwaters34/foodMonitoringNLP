def LD(s, t):
	print s, t
	if s == '':
		return len(t)	
	if t == '':
		return len(s)
	if s[-1] == t[-1]:
		temp = 0
	else:
		temp = 1
	
	res = min(LD(s[:-1], t) + 1,
		LD(s, t[:-1]) + 1,
		LD(s[:-1], t[:-1]) + temp )
	print "Score -> ", s, t, res
	return res 
print LD("python", "peithen")