def wrapStringInHTMLWindows(body, program = 'index', url = 'None'):
    import datetime
    from webbrowser import open_new_tab

    now = datetime.datetime.today().strftime("%Y%m%d-%H%M%S")

    filename = program + '.html'
    f = open(filename,'w')

    wrapper = """<html>
    <head>
    <title>%s output - %s</title>
    </head>
    <body>
        <p>URL: 
            <a href=\"%s\">%s</a></p>
            <mark><p>%s</p> hello everyone</mark>

    </body>
    </html>"""

    whole = wrapper % (program, now, url, url, body)
    f.write(whole)
    f.close()

    open_new_tab(filename)

if __name__ == '__main__':
    wrapStringInHTMLWindows("index", "google.com", "Get the fuck out <br>of here")