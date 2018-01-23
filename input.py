from flask import Flask, flash, Markup
from flask import request 
from flask import render_template
import display_html_2
import pickle 
import sys

def load(fileName):
	with open(fileName, 'r') as f:
		return pickle.load(f)

app = Flask(__name__)
var = load('food_files.pickle')
var.sort()
print(var)
count = -1

@app.route('/')
def my_form():
	try:
		return render_template("input.html")
	except:
		print sys.exc_info()

@app.route('/', methods = ['POST'])
def my_form_post():
	global count 
	global var 
	print len(var)
	text =request.form['text']
	#processed_text = str(var[count])
	count += 1
	count = count%len(var)
	#processed_text
	print var[count]
	try:


		# html_format = read_file(var[count])
		# print "HTNL Format", html_format
		# front_end.wrapStringInHTMLWindows(body=html_format)
		write_to_str, results = display_html_2.read_file(var[count])
		print(results)
		message = Markup(str(write_to_str))
		#print(message)
		return render_template('input.html', filename = var[count],body = message)
	except:
		print('error')
		print sys.exc_info()
		count += 1
		return render_template('input.html', filename = var[count], body = 'Couldnt display')


if __name__ == '__main__':
	app.run()