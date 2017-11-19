#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 17:00:45 2017

@author: kaku
"""

import glob, os, datetime, time, cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import models
from obtain_mistake_rows import excel_to_Dataframe
from plot_learning_curves import acc_loss_visual
from log_writing import log_write

import keras
from keras.models import model_from_json

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def read_path_list(data_path, files, form):
    """
    Read path list of file with 'form'
    
    Parameters
    ----------
    data_path : str
    files : str
    form : str
    
    Returns
    ------
    pathes_list : list
    
    """
    path_list_all = []
    for file in files:
        file_path = os.path.join(data_path, file)
        path_list = sorted(glob.glob(os.path.join(file_path, form)))
        path_list_all.extend(path_list)
    return path_list_all

def read_img_and_labels(value_idx, path_list, *args, diff = False):
    """
    read data and related label from path_list
    Parameters
    ----------
    value_idx : tuple
        beg, end, interval = value_idx
        in order to get value in group
        
    Returns
    -------
    imgs : np.array
        4d
    label s : np.array
        2d
    path_all : list(str)
    """
    
    beg, end, interval = value_idx
    imgs, labels = [], []
    PtNum_idx = np.arange(beg, end, interval).astype(int)
    mean_z_idx = PtNum_idx + 1
    std_z_idx = PtNum_idx + 2
    
    path_all = []
    for idx in path_list:
        
        data = pd.read_csv(idx, header = -1)
        PtNum = data.ix[:,list(PtNum_idx)].values
        mean_z = data.ix[:,list(mean_z_idx)].values
        if diff:
            case_diff = args[0]
            mean_z += case_diff
        std_z = data.ix[:,list(std_z_idx)].values
        label = data.ix[:,1].values
        
        for idx_band in range(len(PtNum)):
            path_all.append(idx)
            PtNum_band = preprocessing.scale(PtNum[idx_band]).reshape(30,-1)
            mean_z_band = preprocessing.scale(mean_z[idx_band]).reshape(30,-1)
            std_z_band = preprocessing.scale(std_z[idx_band]).reshape(30,-1)
            img = np.dstack((PtNum_band, mean_z_band, std_z_band))
            imgs.append(img)
            
        labels.extend(label)
    
    imgs, labels = np.asarray(imgs), (np.asarray(labels)+1)/2
    labels = keras.utils.to_categorical(labels, 2)
    return imgs, labels, path_all

def image_resize(image, region, patch_size):
    """
    Image in different case with different size
    """
    r_min, r_max = region[0], region[1]
    X_small = image[:,r_min:r_max,r_min:r_max,:]
    X_resized_all = []
    for i in range(len(X_small)):
        X_resized =X_small[i] 
        X_resized_all.append(cv2.resize(X_resized,(patch_size, patch_size)))
    image_resized = np.asarray(X_resized_all)
    return image_resized

def save_model(model, model_path, file_time_global):
    """
    Save model into model_path
    """
    json_string=model.to_json()
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    modle_path=os.path.join(model_path,'architecture'+'_'+file_time_global+'.json')
    open(modle_path,'w').write(json_string)
    model.save_weights(os.path.join(model_path,'model_weights'+'_'+ file_time_global+'.h5'),overwrite= True )
    
def read_model(model_path, file_time_global):
    """
    Read model from model_path
    """
    model=model_from_json(open(os.path.join(model_path,'architecture'+'_'+ file_time_global+'.json')).read())
    model.load_weights(os.path.join(model_path,'model_weights'+'_'+file_time_global+'.h5'))
    return model

def result_calculate(model, x_test, y_test, y_pred):
    """
    Calculate result in confusion matrix, acc, and loss
    """
    model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
    score = model.evaluate(x_test, y_test, verbose=0)
    con_mat = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
        
    print('Confusion Matrix:')
    print(str(con_mat[0,0])+'  '+ str(con_mat[0,1]))
    print(str(con_mat[1,0])+'  '+ str(con_mat[1,1]))
    print('Test loss:'+str(score[0]))
    print('Test accuracy:'+ str(score[1]))
    return con_mat, score

def save_result(result_path, y_pred, script_name, model_name, time_global):
    """
    Save result as .npy
    """
    separate_result_file = os.path.join(result_path, time_global)
    if not os.path.isdir(separate_result_file):
        os.mkdir(separate_result_file)
    np.save(os.path.join(separate_result_file,'result_'+script_name+"_"+model_name+'.npy'), y_pred)

def save_result_image(result_path, x_test, y_pred, y_test, path_all_test, time_global):
    """
    Save result images of FN and FP
    """
    separate_result_file = os.path.join(result_path, time_global)
    if not os.path.isdir(separate_result_file):
        os.mkdir(separate_result_file)
    
    y_pred_new = np.argmax(y_pred, axis=1)
    y_test_new = np.argmax(y_test, axis=1)
    
    fn_idx = []
    fn_idx_ori = list(np.clip((y_test_new - y_pred_new),0,1)) 
    [fn_idx.append(index) for index, value in enumerate(fn_idx_ori) if value == 1]
    for idx in range(len(fn_idx)):
        i = fn_idx[idx]
        G2d_name = os.path.basename(path_all_test[i])[:-4]
        Clu_name = str('Clu_') + G2d_name[:6]+'_'+G2d_name[-4:]+'.png'
        Clu_name_path = os.path.dirname(path_all_test[i])+'/'+Clu_name
        Clu_img = plt.imread(Clu_name_path)
        plt.imsave(os.path.join(separate_result_file, 'FN_'+str(idx)+'_'+G2d_name+'.png'), x_test[i])
        plt.imsave(os.path.join(separate_result_file, 'FN_'+str(idx)+'_'+Clu_name),Clu_img)
        
    fp_idx = []
    fp_idx_ori = list(np.clip((y_pred_new - y_test_new),0,1))
    [fp_idx.append(index) for index, value in enumerate(fp_idx_ori) if value == 1]
    for idx in range(len(fp_idx)):
        i = fp_idx[idx]
        G2d_name = os.path.basename(path_all_test[i])[:-4]
        Clu_name = str('Clu_') + G2d_name[:6]+'_'+G2d_name[-4:]+'.png'
        Clu_name_path = os.path.dirname(path_all_test[i])+'/'+Clu_name
        Clu_img = plt.imread(Clu_name_path)
        plt.imsave(os.path.join(separate_result_file, 'FP_'+str(idx)+'_'+G2d_name+'.png'), x_test[i])
        plt.imsave(os.path.join(separate_result_file, 'FP_'+str(idx)+'_'+Clu_name),Clu_img)
    
def rows_rename(data):
    """
    Rename rows' name, eg: from 'Case02_Clu_2000' to 'Case01_G2d_2000.csv'
    Parameters
    ----------
    data : pandas.core.frame.DataFrame
    Returns
    -------
    rows_new_name : renamed lines
    """
    data_values = data.values
    data_list = []
    [data_list.extend(list(data_values[i])) for i in range(len(data_values))]
#    rows = list(pd.read_csv(file).values[:,1])
    rows = data_list    
    rows_new_name = []
    for mistake_row in rows:
        row_new_name = mistake_row[4:-5]+'_G2d_'+mistake_row[-4:]+'.csv'
        rows_new_name.append(row_new_name)
    return rows_new_name

def delete_mistake(path_list, mistake_rows_name):
    """
    Delete mistake rows(with mistake label) in path_list
    """
    del_idx = []
    for i in range(len(path_list)):
        base_name = os.path.basename(path_list[i])
        if base_name in mistake_rows_name:
            del_idx.append(i)
    print('Delete {} images'.format(len(del_idx)))
    for idx in sorted(del_idx, reverse=True):
        del path_list[idx]
    return path_list


def main():
    time_global = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    script_name = os.path.basename(__file__)
    
    data_path = '/media/kaku/HDCL-UT/Shao_pedestrian/data/'
    model_path = '../model/'
    result_path = '../result/'
    #mistake_file = '/media/kaku/Data/Shao_pedestrian/data/mistake_rows.csv'
    mistake_file = '/media/kaku/HDCL-UT/Shao_pedestrian/data/mistake.xlsx'
    
    train_files = ['1-2k','2-3k','3-4k']
    #test_files = ['2-3k', '3-4k', 'Case02']
    #train_files = ['1-2k', '2-3k', '3-4k',]
#    train_files = ['3-4k']
#    test_files = ['2-3k']
    test_files = ['2-3k','3-4k']
#    test_files = ['Sample_Case01_0-1k', 'Sample_Case01_4-5k']
#    test_files = ['Case02']
#    test_files = ['1-2k','2-3k','3-4k']
    form = r'*G2d*.csv'
    value_idx = (7,2707,3)
    case_diff = 3.60 - 2.72
    region = (3,28)
    
    nb_epoch = 200
    patch_size = 30
    batch_size = 32
    cv_ratio = 0.1
    N_Cls = 2
    
    get_model = models.CNN_mnist
    model_name = '2017-11-16-13-19'
    
    # settings
    mistake_line = True #have mistake in some lines
    model_train = False
    model_test = True
    test_label = True
    diff_compensate = False, False #different case between each case, add case_diff
    diff_size = False, False #resize image, make case 2 to be the standard
    
    MODEL_TYPE = get_model.__name__
    
    if mistake_line:
        mistake_data = excel_to_Dataframe(mistake_file)
        mistake_rows_name = rows_rename(mistake_data)
        
    if model_train:
        train_path_list = read_path_list(data_path, train_files, form)
        if mistake_line:
            train_path_list = delete_mistake(train_path_list, mistake_rows_name)
        X, y, path_all_train = read_img_and_labels(value_idx, train_path_list, case_diff, diff=diff_compensate[0])
        if diff_size[0]:
            X = image_resize(X, region, patch_size)
        x_train, x_cv, y_train, y_cv = train_test_split(X, y, test_size=cv_ratio, random_state=42)
        model_name = time_global
        model = get_model(patch_size, N_Cls)
        train_begin = time.time()
        History = model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(x_cv, y_cv))
        time_train = (time.time()-train_begin)
        save_model(model, model_path, model_name)
        acc_loss_visual(History.history, result_path, script_name, model_name, time_global)
           
    else:
        model = read_model(model_path, model_name)
    
    if model_test:
        test_path_list = read_path_list(data_path, test_files, form)    
        if mistake_line:
            test_path_list = delete_mistake(test_path_list, mistake_rows_name)
        x_test, y_test, path_all_test = read_img_and_labels(value_idx, test_path_list,case_diff, diff=diff_compensate[1])
        if diff_size[1]:
            x_test = image_resize(x_test, region, patch_size)
        test_begin = time.time()
        y_pred = model.predict(x_test)
        time_test = (time.time()-test_begin)
        if test_label:
            con_mat, score = result_calculate(model, x_test, y_test, y_pred)
            save_result(result_path, y_pred, script_name, model_name, time_global)
            save_result_image(result_path, x_test, y_pred, y_test, path_all_test, time_global)
        else:
            print('Without labels')
    
    settings = [train_files,test_files,diff_compensate,diff_size,region]
    if model_train and model_test:
        if test_label:
            log_write(settings, result_path, time_global, script_name,  patch_size, N_Cls, batch_size, \
                          nb_epoch, cv_ratio, model, model_name, MODEL_TYPE,\
                          y, time_train, History, len(y_test),time_test, con_mat,score,\
                          train_mode = True, test_mode = True, label_mode = True)
        else:
            log_write(settings, result_path, time_global, script_name,  patch_size, N_Cls, batch_size, \
                          nb_epoch, cv_ratio, model, model_name, MODEL_TYPE,\
                          y, time_train, History, len(y_test),time_test, \
                          train_mode = True, test_mode = True)
    
    elif model_train and not model_test:
            log_write(settings, result_path, time_global, script_name,  patch_size, N_Cls, batch_size, \
                          nb_epoch, cv_ratio, model, model_name, MODEL_TYPE,\
                          y, time_train, History,\
                          train_mode = True)
    else:
        if test_label:
            log_write(settings, result_path, time_global, script_name,  patch_size, N_Cls, batch_size, \
                          nb_epoch, cv_ratio, model, model_name, MODEL_TYPE,\
                          0,0,0,len(y_test),time_test,con_mat,score,\
                          test_mode = True, label_mode = True)
        else:
            log_write(settings, result_path, time_global, script_name,  patch_size, N_Cls, batch_size, \
                          nb_epoch, cv_ratio, model, model_name, MODEL_TYPE,\
                          0,0,0,len(y_test),time_test,\
                          test_mode = True)      
    
if __name__ == '__main__':
    print('Start Pedestrain Detection')
    main()
else:
    print('Hello')
