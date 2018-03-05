from display_html_old import read_file

fileName = 'HSLLD/HV1/MT/admmt1.cha'
html_format, results = read_file(fileName, base_accuracy_on_how_many_unique_food_items_detected=True, wordnet_setting='all', use_pos_tags_for_wordnet=False)
print(results)


