from data_loader import *

l= ["00", "03", "06", "09", "12", "15", "18", "21"]
thresholds = [8,10,12,15,18,25,30]
for t in l: 
    #filename = '/ourdisk/hpc/geocarb/vishnupk/folds/xiao_data_' + t + '.pkl'
    filename = '/home/vishnu/Documents/geocarb/methane/pkl/xiao_data_' + t + '.pkl'
    
    for threshold in thresholds:
        data_loader(filename = filename, test_size = 0.2, random_state = 42,threshold = threshold, repeat = False, data_path = '/home/vishnu/Documents/geocarb/methane/' + t + '/' + str(threshold) , testing = True)
        print("Threshold: ", threshold, " done")