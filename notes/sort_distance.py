import pickle


class sorting():
    def __init__(self, fileName):
        temp_array = []
        f = open(fileName, "r").read().split('\n')
        for lines in f:
            lines = lines.split(',')
            #[x+1 if x >= 45 else x+5 for x in l]
            lines = [round(float(lines[x]), 3) if x >1 else lines[x] for x in xrange(len(lines))]
            print len(lines)
            if len(lines) == 5:
                temp_array.append(lines)
            #print lines
        print temp_array
        #Sorting 
        temp_2_array = sorted(temp_array, key = lambda x: x[4])
        print temp_2_array

        self.write_2_file(temp_2_array, "sort_distance_customized_results.csv")

        # self.write_2_file(temp_2_array, "sort_distance_edit_distance_3.csv")

        #print f
    
    def write_2_file(self, variable, fileName):
        temp_text = ''
        for lines in variable:
            temp = ''
            for words in lines:
                temp += str(words) + ","
            temp = temp[:-1]
            temp += "\r\n"
            temp_text += temp

        with open(fileName, "w") as f:
            #pickle.dump(temp_text, f)
            f.write(temp_text)

if __name__ == '__main__':

    sort = sorting("edit_distance_customized.txt")

    # sort = sorting("edit_distance_3.txt")

