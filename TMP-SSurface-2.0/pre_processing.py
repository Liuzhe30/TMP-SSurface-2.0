import numpy as np
import os
from keras.utils import to_categorical
import argparse

def format(f, n):
    if round(f) == f:
        m = len(str(f)) - 1 - n
        if f / (10 ** m) == 0.0:
            return f
        else:
            return float(int(f) / (10 ** m) * (10 ** m))
    return round(f, n - len(str(int(f)))) if len(str(f)) > n+1 else f

dict = {'C':0, 'D':1, 'S':2, 'Q':3, 'K':4,
        'I':5, 'P':6, 'T':7, 'F':8, 'N':9,
        'G':10, 'H':11, 'L':12, 'R':13, 'W':14,
        'A':15, 'V':16, 'E':17, 'Y':18, 'M':19}
        
class Processor:
    
    def data_pre_processing(self, fasta, pssm_path, window_length):
        train_fasta = open(fasta)
        line = train_fasta.readline()
        pdb_id = ""
        x_train = []
        y_train = []    
        while line:
            codelist = []
            if(line[0] == ">"):
                pdb_id = line[1:].strip().split()[0]
                line = train_fasta.readline()
                continue
            #print(pdb_id)
            seq_length = len(line) - 1
            '''
            #-------- used for feature without one-hot code ----------#
            
            t = int((window_length - 1) / 2)
            code = np.zeros(seq_length, int)
            code = np.r_[np.ones(t, int), code]        
            code = np.r_[code, np.ones(t, int)]            
            
            #-------- used for feature without one-hot code ----------#
            '''
            
            #-------- one-hot encoded (len * 20) ----------#
            for i in line:
                if(i != "\n"):
                    code = dict[i.upper()]
                    codelist.append(code)
            data = np.array(codelist)
            #print('Shape of data (BEFORE encode): %s' % str(data.shape))
            encoded = to_categorical(data)
            if(encoded.shape[1] < 20):
                column = np.zeros([encoded.shape[0], 20 - encoded.shape[1]], int)
                encoded = np.c_[encoded, column]
            #print('Shape of data (AFTER  encode): %s\n' % str(encoded.shape))
            #-------- one-hot encoded (len * 20) ----------#
            
            #-------- noseq encoded (len + window_length - 1) * 21 ----------#
            code = encoded
            length = code.shape[0]
            noSeq = np.zeros([length, 1], int)
            code = np.c_[code, noSeq]
            
            t = int((window_length - 1) / 2)
            
            code = np.r_[np.c_[np.zeros([t, 20], int), np.ones(t, int)], code]        
            code = np.r_[code, np.c_[np.zeros([t, 20], int), np.ones(t, int)]]
            #-------- noseq encoded (len + window_length - 1) * 21 ----------#
            
            '''
            #-------- add length of the sequence to (len + window_length - 1) * 22 ----------#
            length = code.shape[0]
            code = np.c_[code, np.ones([length, 1], int) * seq_length]
            #-------- add length of the sequence to (len + window_length - 1) * 22 ----------#
            '''
            
            '''
            #-------- add distribution of the residue ----------#
            length = code.shape[0]
            list_residue = ['C', 'D', 'S', 'Q', 'K', 'I', 'P', 'T', 'F', 'N', 'G', 'H', 'L', 'R', 'W', 'A', 'V', 'E', 'Y', 'M']
            for residue in list_residue:
                cnt = line.count(residue)
                code = np.c_[code, np.ones([length, 1], int) * cnt]
            #-------- add distribution of the residue ----------#    
            '''
            
            #-------- add pssm feature ----------# 
            length = code.shape[0]
            list_dir = os.getcwd()
            pssm_path = pssm_path + pdb_id + ".pssm"
            pssm_file = os.path.join(list_dir, pssm_path)
            if(os.path.exists(pssm_file)):
                pssm = open(pssm_file)
                pssm_matrix = np.zeros([length, 20], int)
                filelines = pssm.readlines()
                top = t - 1
                for pssm_line in filelines:
                    if(len(pssm_line.split()) == 44):
                        each_item = pssm_line.split()[2:22]
                        #print(each_item)
                        for j in range(0, 20):
                            if(top == length - 1 - t):
                                break
                            try:
                                pssm_matrix[top, j] = int(each_item[j])
                            except IndexError:
                                pass
                        top += 1
                code = np.c_[code, pssm_matrix]        
            else:
                code = np.c_[code, np.zeros([length, 20], int)]
                print(str(pdb_id) + " not found!!")
            #-------- add pssm feature ----------# 
            
            '''
            #-------- add noseq feature ----------#
            length = code.shape[0] 
            t = int((window_length - 1) / 2)
            noSeq = np.zeros([length - window_length + 1, 1], int)
            noSeq = np.r_[np.ones([t, 1], int), noSeq]
            noSeq = np.r_[noSeq, np.ones([t, 1], int)]
            code = np.c_[code, noSeq]
            #-------- add noseq feature ----------#
            '''
            
            #-------- sliding window (window_length * feature) ---------#
            length = code.shape[0]
            feature = code.shape[1]
            top = 0
            buttom = window_length
            while(buttom <= length):
                x_train.append(code[top:buttom])            
                top += 1
                buttom += 1
            #print(len(window_list))
            #-------- sliding window (window_length * feature) ---------#
            
            line = train_fasta.readline()    
            
        x_train = np.array(x_train)
        print(x_train.shape)  
        np.save("temp.npy", x_train)      
        
if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--fasta', default='sample/sample.fasta')
    parser.add_argument('--pssm_path', default='sample/pssm/')
    args = parser.parse_args()
    #print(args)    
    
    processor = Processor()
    window_length = 19
    fasta = args.fasta
    pssm_path = args.pssm_path
    
    processor.data_pre_processing(fasta, pssm_path, window_length) 
    