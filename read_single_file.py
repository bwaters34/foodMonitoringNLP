import display_html_2

fileName = 'HSLLD/HV1/MT/admmt1.cha'
html_format, results = display_html_2.read_file(fileName, base_accuracy_on_how_many_unique_food_items_detected=True)
print(results)