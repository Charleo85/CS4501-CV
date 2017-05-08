#referenced from scribe and other IAM database preprocess
import numpy as np
import math
import random
import os
import pickle
import xml.etree.ElementTree as ET

from utils import *

class DataLoader():
    def __init__(self, data_dir, alphabet, batch_size, tsteps, data_scale, tsteps_per_ascii, limit = 500):
        self.data_dir = data_dir
        self.alphabet = alphabet
        self.batch_size = batch_size
        self.tsteps = tsteps
        self.data_scale = data_scale
        self.ascii_steps = tsteps//tsteps_per_ascii
        self.limit = limit

        train_file = os.path.join(self.data_dir, "strokes_training_data.cpkl")
        stroke_dir = self.data_dir + "/stroke"
        text_dir = self.data_dir + "/text"

        if not (os.path.exists(train_file)) :
            self.preprocess(stroke_dir, text_dir, train_file)
        else :
            print("Training data found, proceed to loading")

        self.load(train_file)
        self.reset_batch_pointer()

    def getStrokes(filename):
            tree = ET.parse(filename)
            root = tree.getroot()
            result = []
            x_offset = 1e20
            y_offset = 1e20
            y_height = 0
            for i in range(1, 4):
                x_offset = min(x_offset, float(root[0][i].attrib['x']))
                y_offset = min(y_offset, float(root[0][i].attrib['y']))
                y_height = max(y_height, float(root[0][i].attrib['y']))
            y_height -= y_offset
            x_offset -= 100
            y_offset -= 100
            for stroke in root[1].findall('Stroke'):
                points = []
                for point in stroke.findall('Point'):
                    points.append([float(point.attrib['x'])-x_offset,float(point.attrib['y'])-y_offset])
                result.append(points)
            return result
    
    def getText(filename, line_number):
            with open(filename, "r") as f:
                s = f.read()
            s = s[s.find("CSR"):]
            if len(s.split("\n")) > line_number+2:
                s = s.split("\n")[line_number+2]
                return s
            else:
                return ""
    
    def convert_stroke_to_array(stroke):
            n_point = 0
            for i in range(len(stroke)):
                n_point += len(stroke[i])
            stroke_data = np.zeros((n_point, 3), dtype=np.int16)

            prev_x = 0
            prev_y = 0
            counter = 0

            for j in range(len(stroke)):
                for k in range(len(stroke[j])):
                    stroke_data[counter, 0] = int(stroke[j][k][0]) - prev_x
                    stroke_data[counter, 1] = int(stroke[j][k][1]) - prev_y
                    prev_x = int(stroke[j][k][0])
                    prev_y = int(stroke[j][k][1])
                    stroke_data[counter, 2] = 0
                    if (k == (len(stroke[j])-1)): # end of stroke
                        stroke_data[counter, 2] = 1
                    counter += 1
            return stroke_data
        
    def preprocess(self, stroke_dir, ascii_dir, data_file):
        # create data file from raw xml files from iam handwriting source.
        
        # build the list of xml files
        filelist = []
        # Set the directory you want to start from
        rootDir = stroke_dir
        for dirName, subdirList, fileList in os.walk(rootDir):
            for fname in fileList:
                filelist.append(dirName+"/"+fname)

        # build stroke database of every xml file inside iam database
        strokes = []
        asciis = []
        for i in range(len(filelist)):
            if (filelist[i][-3:] == 'xml'):
                stroke_file = filelist[i]
#                 print("processing"+str(stroke_file))
                stroke = convert_stroke_to_array(getStrokes(stroke_file))
                
                ascii_file = stroke_file.replace("lineStrokes","ascii")[:-7] + ".txt"
                line_number = stroke_file[-6:-4]
                line_number = int(line_number) - 1
                text = getText(ascii_file, line_number)
                strokes.append(stroke)
                asciis.append(ascii)
        
        f = open(data_file,"wb")
        pickle.dump([strokes,asciis], f, protocol=2)
        f.close()
        print(len(strokes) + " lines have been found")


    def load(self, data_file):
        f = open(data_file,"rb")
        [self.raw_stroke_data, self.raw_ascii_data] = pickle.load(f, encoding='latin1')
        f.close()

        # goes thru the list, and only keeps the text entries that have more than tsteps points
        self.stroke_data = []
        self.ascii_data = []
        self.valid_stroke_data = []
        self.valid_ascii_data = []
        counter = 0

        # every 1 in 20 (5%) will be used for validation data
        cur_data_counter = 0
        for i in range(len(self.raw_stroke_data)):
            data = self.raw_stroke_data[i]
            if len(data) > (self.tsteps+2):
                # removes large gaps from the data
                data = np.minimum(data, self.limit)
                data = np.maximum(data, -self.limit)
                data = np.array(data,dtype=np.float32)
                data[:,0:2] /= self.data_scale
                cur_data_counter = cur_data_counter + 1
                if cur_data_counter % 20 == 0:
                  self.valid_stroke_data.append(data)
                  self.valid_ascii_data.append(self.raw_ascii_data[i])
                else:
                    self.stroke_data.append(data)
                    self.ascii_data.append(self.raw_ascii_data[i])

        # minus 1, since we want the ydata to be a shifted version of x data
        self.num_batches = int(len(self.stroke_data) / self.batch_size)
        print("\tloaded dataset:")
        print("\t\t{} train individual data points".format(len(self.stroke_data)))
        print("\t\t{} valid individual data points".format(len(self.valid_stroke_data)))
        print("\t\t{} batches".format(self.num_batches))

    def validation_data(self):
        # returns validation data
        x_batch = []
        y_batch = []
        ascii_list = []
        for i in range(self.batch_size):
            valid_ix = i%len(self.valid_stroke_data)
            data = self.valid_stroke_data[valid_ix]
            x_batch.append(np.copy(data[:self.tsteps]))
            y_batch.append(np.copy(data[1:self.tsteps+1]))
            ascii_list.append(self.valid_ascii_data[valid_ix])
        one_hots = [to_one_hot_train(s, self.ascii_steps, self.alphabet) for s in ascii_list]
        return x_batch, y_batch, ascii_list, one_hots

    def next_batch(self):
        # returns a randomized, tsteps-sized portion of the training data
        x_batch = []
        y_batch = []
        ascii_list = []
        for i in range(self.batch_size):
            data = self.stroke_data[self.idx_perm[self.pointer]]
            idx = random.randint(0, len(data)-self.tsteps-2)
            x_batch.append(np.copy(data[:self.tsteps]))
            y_batch.append(np.copy(data[1:self.tsteps+1]))
            ascii_list.append(self.ascii_data[self.idx_perm[self.pointer]])
            self.tick_batch_pointer()
        one_hots = [to_one_hot_train(s, self.ascii_steps, self.alphabet) for s in ascii_list]
        return x_batch, y_batch, ascii_list, one_hots

    def tick_batch_pointer(self):
        self.pointer += 1
        if (self.pointer >= len(self.stroke_data)):
            self.reset_batch_pointer()
    def reset_batch_pointer(self):
        self.idx_perm = np.random.permutation(len(self.stroke_data))
        self.pointer = 0

# convert text into one hot vector
def to_one_hot(text, text_length, alphabet):
    v = np.zeros((text_length,1 + len(alphabet)))
    counter = 0
    for char in text:
        v[counter][alphabet.find(char) + 1] = 1
        counter += 1
    return v

def to_one_hot_train(s, ascii_steps, alphabet):
    steplimit=3e3; s = s[:3e3] if len(s) > 3e3 else s # clip super-long strings
    seq = [alphabet.find(char) + 1 for char in s]
    if len(seq) >= ascii_steps:
        seq = seq[:ascii_steps]
    else:
        seq = seq + [0]*(ascii_steps - len(seq))
    one_hot = np.zeros((ascii_steps,len(alphabet)+1))
    one_hot[np.arange(ascii_steps),seq] = 1
    return one_hot
