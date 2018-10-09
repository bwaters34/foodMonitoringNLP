import pickle
import numpy as np

class fileStats:
    def __init__(self, pickle_fileName):
        self.fileName = pickle_fileName

    def open_file(self, fileName = None):
        if fileName is None: fileName = self.fileName
        with open(fileName, 'r') as f:
            return pickle.load(f)

    def fileStats(self, fileName = None):
        count = {}
        if fileName is None: fileName = self.fileName
        for f in fileName:
            # print(f)
            f = '/'.join(f.split('/')[1:])
            # print(f)
            transcript = open(f, 'r').read()[:-1]
            count[f] = len([line for line in transcript.split('\n') if line[0] == '*'])
        print(count)
        return count.values()

    def calculate_min_max(self, stats_list):
        stats_list = np.asarray(stats_list, 'float64')
        print(stats_list)
        min = np.amin(stats_list)
        max = np.amax(stats_list)
        std = np.std(stats_list)
        mean = np.mean(stats_list)
        print(min, max, std, mean)

if __name__ == '__main__':
    fs = fileStats('food_files_solution.pickle')
    # fs = fileStats('food_files.pickle')
    fileNames = fs.open_file()
    # print (fileNames)
    stats = fs.fileStats(fileNames)
    fs.calculate_min_max(stats)
