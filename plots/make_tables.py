import pandas as pd
import numpy as np
import torch


def make_table():
    dataset = 'DCOPF'
    model_ids = ['511_DCOPF_primal_obj_s', '511_DCOPF_primal_soft_s', '511_DCOPF_nn_pretrain_s']
    out_name = 'DCOPF2213'

    table_columns = model_ids
    table_rows = ['test_optimality_gap_mean',
                     'test_optimality_gap_worst',
                     'test_eq_mean',
                     'test_eq_max',
                     'test_eq_worst',
                     'test_scaled_eq_mean',
                     'test_scaled_eq_max',
                     'test_scaled_eq_worst',
                     'test_ineq_mean',
                     'test_ineq_max',
                     'test_ineq_worst',
                     'train_time',
                     'val_time',
                     'test_time',
                     'val_proj',
                     'test_proj',]
    columns_list = []

    for model_id in model_ids:
        csv_path = f'../logs/{model_id}.csv'
        df = pd.read_csv(csv_path, header=None)
        df_dict = df.set_index(0).T.to_dict('list')
        column_data = {'model_id': model_id}
        column_data.update({row: df_dict[row][0] for row in table_rows})
        columns_list.append(column_data)

    table_df = pd.concat([pd.DataFrame([column]) for column in columns_list], ignore_index=True)
    table_df.set_index('model_id', inplace=True)
    table_df = table_df.transpose()
    table_df.columns = table_df.columns.str.replace('_', ' ')
    table_df.index = table_df.index.str.replace('_', ' ')
    table_df.to_csv(f'./{out_name}_table.csv')

