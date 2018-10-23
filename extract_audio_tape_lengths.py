import os
import datetime
import csv

path_base = "HSLLD"
corpus = ["HV1", "HV2", "HV3", "HV5", "HV7"]
ending = "MT"

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
for corp in corpus:
    dir_path = "/".join([path_base, corp, ending])
    for file_name in os.listdir(dir_path):
        file_path = dir_path + '/' + file_name
        file_paths.append(file_path)

#
# #  -- JUST SOLUTIONS --
# for ending in solution_file_name_endings:
#     file_path = "HSLLD/HV1/MT/" + ending
#     file_paths.append(file_path)

for file_path in file_paths:
    with open(file_path) as f:
        lines = f.readlines()
        comments = []
        for line in lines:
            if line.startswith("@Comment:\tTime is"):
                comments.append(line)
        if len(comments) == 0:
            print("comment not found: {}".format(file_path))
            not_found.append(file_path)
        else:
            # if len(comments) > 1:
            #     two_comments.append((file_path, comments))
            # # print(lines[-2])
            two_comments = []
            for line in comments:
                time_str = (line.split()[-1])
                try:
                    minute_str, second_str = time_str.split(':')
                    minute = int(minute_str)
                    second = int(second_str)
                    file_time = datetime.timedelta(seconds=second, minutes=minute)
                    two_comments.append(file_time)
                    # times_and_files.append((file_time, file_path))
                except ValueError:
                    print("ValueError")
                    print(line)
            if len(two_comments) < 2:
                print(file_path)
                print(comments)
                print(two_comments)
                not_two.append((two_comments, file_path))
            else:
                times_and_files.append((file_path, two_comments))
print("number of files with times:")
print(len(times_and_files))
times = zip(*times_and_files)[1]
print(times[0])
seconds = []
for time_range in times:
    end = time_range[-1]
    start = time_range[0]  #chuck out all times in the middle
    delta = end - start
    seconds.append(delta.total_seconds())
# seconds = [(x[0]-x[1]).total_seconds() for x in times]
#
# times = zip(*times_and_files)[0]
# print(times)
# seconds = [t.total_seconds() for t in times]
print(seconds)
more_than_30 = [x for x in seconds if x > 60*24]
print(len(more_than_30))
print("More than 24 mins")
avg = sum(more_than_30)/float(len(more_than_30))
print(avg)
print("average")
print(avg / 60)
print(max(seconds))
i = seconds.index(max(seconds))
# print(times_and_files[i])
print(min(seconds))
i = seconds.index(min(seconds))
# print(times_and_files[i])


# for x in two_comments:
#     print(x[0])
#     print(x[1])
# print(len(two_comments))


print("not found:")
print(not_found)
print(len(not_found))
print("not two comments:")
print(not_two)
print(len(not_two))

with open("extract_file_names_results.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["fileName", "total seconds", "comments"])
    for i in range(len(times_and_files)):
        file_name = times_and_files[i][0]
        time_split = int(seconds[i])
        row = [file_name, time_split, '']
        writer.writerow(row)
        # f.write("{},{},{}:{},\n".format(file_name, time_split, time_split/60, time_split%60))
        # print("{},{}".format(file_name, time_split))
    for comments, file_path in not_two:
        row = [file_path, "", "time not found:"]
        writer.writerow(row)

