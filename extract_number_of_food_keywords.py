import os
import datetime
import csv
import statistics
times_and_files = []
not_two = []
not_found = []

solution_file_name_endings = """admmt1.cha
aimmt1.cha
allmt1.cha
anamt1.cha
annmt1.cha
aprmt1.cha
bramt1.cha
brimt1.cha
brnmt1.cha
brtmt1.cha
casmt1.cha
conmt1.cha
davmt1.cha
diamt1.cha
emimt1.cha
ethmt1.cha
geomt1.cha
gilmt1.cha
gremt1.cha
guymt1.cha
jacmt1.cha""".split()

file_paths = []
#  -- ALL FILES --
# for corp in corpus:
#     dir_path = "/".join([path_base, corp, ending])
#     for file_name in os.listdir(dir_path):
#         file_path = dir_path + '/' + file_name
#         file_paths.append(file_path)


#  -- JUST SOLUTIONS --
for ending in solution_file_name_endings:
    file_path = "HSLLD/HV1/MT/" + ending
    file_paths.append(file_path)

num_lines = []
for file_path in file_paths:
    with open(file_path) as f:
        lines = f.readlines()
        just_text = 0
        for line in lines:
            if line[0] == "*":
                just_text += 1
        num_lines.append(just_text)
print(num_lines)
print("max lines: {}".format(max(num_lines)))
print("mean lines: {}".format(sum(num_lines) / len(num_lines)))
print("min lines: {}".format(min(num_lines)))
print("std.dev: {}".format(statistics.stdev(num_lines)))