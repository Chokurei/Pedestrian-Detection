#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 15:57:05 2017

@author: kaku

Use to delete rows with mistake

"""
import os
import pandas as pd

#excel_path = '/media/kaku/Data/Shao_pedestrian/data/mistake.xlsx'
    
def excel_to_Dataframe(excel_path):
    """
    Get mistake lines from excel and change into Dataframe, save result csv file in excel_path.
    Parameters
    ----------
    excel_path : mistake rows file
    Returns
    -------
    new_df : pandas.core.frame.DataFrame
        new file in which mistake rows are deleted
    
    Examples
    --------
    rows in excel like:
        1. in some lines like 'Clu_Case01_3780至Clu_Case01_3839'
        2. in some lines only like 'Clu_Case01_3780'
    """
    df = pd.read_excel(excel_path, header = None).values
    df_rows = []
    # change into list in char
    for idx_row in range(len(df)):
        df_list = list(list(df[idx_row])[0])
        df_rows.append(df_list)
        
    df_rows_new = []
    # in some rows mistake refer like 'Clu_Case01_3780至Clu_Case01_3839', need to expand the dataset
    for i in range(len(df_rows)):
        df_row = df_rows[i]
        if '至' in df_row:
            zhi_idx = df_row.index('至')
            df_row_part_1 = df_row[:zhi_idx]
            min_idx = ''.join(df_row_part_1)[-4:]
            df_row_part_2 = df_row[zhi_idx+1:]
            max_idx = ''.join(df_row_part_2)[-4:]
            df_row_basic = df_row_part_1[:-4]
            for idx_add in range(0,int(max_idx)-int(min_idx)+1):
                added_idx = str(int(min_idx) + idx_add)
                if len(list(added_idx)) == 3:
                    added_idx = ['0'] + added_idx
                df_row = ''.join(df_row_basic) + added_idx
                df_rows_new.append(df_row)
        else:
            df_row = ''.join(df_row)
            df_rows_new.append(df_row)
    
    new_df = pd.DataFrame(data = df_rows_new, index = None)
    return new_df

if __name__ == '__main__':
    excel_path = '/media/kaku/Data/Shao_pedestrian/data/mistake.xlsx'
    new_df = excel_to_Dataframe(excel_path)
    new_df.to_csv(os.path.join(os.path.dirname(excel_path), 'mistake_rows.csv'))
    print('Save csv file in:' + excel_path)
else:
    print("function : 'excel_to_Dataframe' can be used")    
    