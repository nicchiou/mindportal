import time
import os
import pickle5 as pickle

import numpy as np
import torch
from spatiotemporal_cnn.dataset_bci import DatasetBuilder, SingleSubjectData
from spatiotemporal_cnn.models_bci import SpatiotemporalCNN
from torch.utils.data import DataLoader
from utils import constants


def time_inference(model, device, data_loader):

    model.eval()
    pred = list()
    times = list()
    with torch.no_grad():
        for _, data, labels in data_loader:

            t0 = time.time()
            data = data.to(device) \
                if isinstance(data, torch.Tensor) \
                else [i.to(device) for i in data]
            labels = labels.to(device)

            outputs = model(data).squeeze()
            probabilities = torch.sigmoid(outputs)
            predicted = probabilities > 0.5
            t1 = time.time()
            pred.append(predicted.data.int())
            times.append(1000 * (t1 - t0))  # time in ms

    return times


results_dir = '/shared/rsaas/nschiou2/mindportal/bci/mot/motor_LR/'\
              'spatiotemporal_cnn/voxel_space/max_abs_scale/'\
              'baseline_ph_viable_vox/hyperopt/s_2804/hyperopt_search/'
model_dir = os.path.join(
    results_dir,
    'train_with_configs_fff29d68_10_D=9,F1=16,F2=7,T=75,batch_size=32,'
    'dropout=0.5,fs=52,l2=0.0001,lr=0.001_2022-06-08_11-49-44')
data_dir = os.path.join(
    constants.PH_SUBJECTS_DIR, 'bci', 'voxel_space',
    'avg_rl_cropped_00_12')

subject = '2804'
train_submontages = ['abc']


if __name__ == '__main__':

    bci_data = SingleSubjectData(
        data_dir=data_dir,
        subject_id=subject,
        train_submontages=train_submontages,
        classification_task='motor_LR',
        expt_type='mot',
        filter_zeros=True,
        input_space='voxel_space',
        data_type='ph'
    )

    num_features = bci_data.get_num_viable_features()

    db = DatasetBuilder(
        data=bci_data, seed=42, seed_cv=15,
        max_abs_scale=True,
        impute='zero')

    with open(os.path.join(model_dir, 'params.pkl'), 'rb') as pickle_file:
        model_p = pickle.load(pickle_file)
    model_p['C'] = num_features

    model = SpatiotemporalCNN(
        C=model_p['C'],
        F1=model_p['F1'],
        D=model_p['D'],
        F2=model_p['F2'],
        p=model_p['dropout'],
        fs=model_p['fs'],
        T=model_p['T']
    )

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
    model = model.to(device)

    times = list()
    for i, (inner_train_valids, test_dataset) in enumerate(
            db.build_datasets(cv=5, nested_cv=1)):

        for j, (train_dataset, valid_dataset) in enumerate(
                inner_train_valids):

            test_dataset.impute_chan(train_dataset)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

            times.extend(time_inference(model, device, test_loader))

    print(f'Average length of inference per example: {np.mean(times):.3f}ms')
