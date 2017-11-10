#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 23:13:39 2017

@author: kaku
"""
import os, sys
import numpy as np

def log_write(result_path, time_global, script_name, PATCH_SIZE, N_ClASS, BATCH_SIZE,\
              EPOCH, CV_RATIO, model, model_name, model_type,\
              *args,\
              train_mode = False, test_mode = False, label_mode = False):
    stdout = sys.stdout
    
    log_file=open(os.path.join(result_path,'my_log.txt'),'a')
    
    sys.stdout = log_file
    
    print('########################Time: '+time_global+'########################')
    print('############################File: '+script_name+'########################')
    if train_mode:
        print('Model type: {}'.format(model_type))
        print('Training sample size: '+''+str(PATCH_SIZE)+' x '+str(PATCH_SIZE))
        TRINING_SAMPLES = args[0]
        time_train = args[1]
        History = args[2]
        
        num_pos = list(np.argmax(TRINING_SAMPLES, axis = 1)).count(1)
        print('Number of trianing samples: '+str(int(len(TRINING_SAMPLES) * (1-CV_RATIO))))
        print('          viladation samples: '+str(int(len(TRINING_SAMPLES) * CV_RATIO)))    
        print('          positive samples: '+str(num_pos)+'  negative samples:'+str(len(TRINING_SAMPLES) - num_pos))    
        print('Batch_size: '+str(BATCH_SIZE))
        print('Iteration: '+str(EPOCH))
        print('Training_time: '+str(time_train)+'    Every_iter:'+str((time_train)/EPOCH))
        print('Training:')
        print('         accuracy: ' + str(History.history['acc'][-1])+'     loss: '+str(History.history['loss'][-1]))
        print('Validation:')
        print('         accuracy: ' + str(History.history['val_acc'][-1])+'     loss: '+str(History.history['val_loss'][-1]))
        print("\n")
    else:
        print('Using model: {}'.format(model_name))
        
    if test_mode:
#        print('Load model: {}'.format(model_name))
        test_amount = args[3]
        test_time = args[4]
        print('Testing image pieces: '+str(test_amount))
        print('Testing time: '+ str(test_time))
        if label_mode:
            con_mat = args[5]
            score = args[6]
            
            
            print("Testing result:")
            print('     confusion matrix: ')
            print('                      '+str(con_mat[0,0])+'  '+ str(con_mat[0,1]))
            print('                      '+str(con_mat[1,0])+'  '+ str(con_mat[1,1]))

            print('     Test loss:'+str(score[0]))
            print('     Test accuracy:'+str(score[1]))
        else:            
            print('Test image without ground truth')
            print("\n")            
        
    if train_mode:

        print("Model details:")
        model.summary()

    print("\n")
    
    sys.stdout = stdout
    log_file.close()  
    
if __name__ == '__main__':
    print('Hello')
else:
    print("function : 'log_write' can be used")