import pandas as pd
import numpy as np
import json
import torch


def make_table():
    dataset = 'DCOPF'
    model_ids = ['511_DCOPF_primal_obj_none', '511_DCOPF_primal_obj_pock', '511_DCOPF_primal_obj_ruiz']
    out_name = 'give_me_a_name'

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


def make_projection_summary():
    out_name = 'give_me_a_name'

    projections = ['POCS', 'EAPM']
    preconditions = ['none', 'Pock-Chambolle', 'Ruiz']
    runs = range(5)

    summary_df = pd.DataFrame(columns=['Projection', 'Precondition',
                                       'Avg Proj Num Train Mean', 'Avg Proj Num Train Std',
                                       'Avg Proj Num Val Mean', 'Avg Proj Num Val Std',
                                       'Unconverged Rate Train Mean','Unconverged Rate Train Std',
                                       'Unconverged Rate Val Mean', 'Unconverged Rate Val Std'])

    for projection in projections:
        for precondition in preconditions:
            avg_proj_num_train_list = []
            avg_proj_num_val_list = []
            unconverged_rate_train_list = []
            unconverged_rate_val_list = []
            for run in runs:
                csv_path = f'../data/sanity_check/{projection}_{precondition}{run}.csv'
                df = pd.read_csv(csv_path, header=None)
                df_dict = df.set_index(0).T.to_dict('list')
                avg_proj_num_train_list.append(df_dict['avg proj num train'][0])
                avg_proj_num_val_list.append(df_dict['avg proj num val'][0])
                unconverged_rate_train_list.append(df_dict['unconverged rate train'][0])
                unconverged_rate_val_list.append(df_dict['unconverged rate val'][0])

            avg_proj_num_train_mean = np.mean(avg_proj_num_train_list)
            avg_proj_num_train_std = np.std(avg_proj_num_train_list)
            avg_proj_num_val_mean = np.mean(avg_proj_num_val_list)
            avg_proj_num_val_std = np.std(avg_proj_num_val_list)
            unconverged_rate_train_mean = np.mean(unconverged_rate_train_list)
            unconverged_rate_train_std = np.std(unconverged_rate_train_list)
            unconverged_rate_val_mean = np.mean(unconverged_rate_val_list)
            unconverged_rate_val_std = np.std(unconverged_rate_val_list)

            summary_df = summary_df.append({
                'Projection': projection,
                'Precondition': precondition,
                'Avg Proj Num Train Mean': avg_proj_num_train_mean,
                'Avg Proj Num Train Std': avg_proj_num_train_std,
                'Avg Proj Num Val Mean': avg_proj_num_val_mean,
                'Avg Proj Num Val Std': avg_proj_num_val_std,
                'Unconverged Rate Train Mean': unconverged_rate_train_mean,
                'Unconverged Rate Train Std': unconverged_rate_train_std,
                'Unconverged Rate Val Mean': unconverged_rate_val_mean,
                'Unconverged Rate Val Std': unconverged_rate_val_std
            }, ignore_index=True)

            summary_df.to_csv(f'./data/{out_name}_sanity_check.csv', index=False)

