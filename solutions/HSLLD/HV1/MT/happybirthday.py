a = "104 097 112 112 121 032 098 105 114 116 104 100 097 121 033"
output = ""
for x in a.split():
    output += chr(int(x))
print(output)