import time
from cnn_model import *
from audio_data import CNNDataset
from cnn_training import *
import argparse
import torch
import torch.nn as nn
import hdf5storage
import os


def main(config):
    dtype = torch.FloatTensor
    ltype = torch.LongTensor

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print('Using CUDA.')
        dtype = torch.cuda.FloatTensor
        ltype = torch.cuda.LongTensor

    optimizer = None
    if config.step!='features':
        model = CNNModel(kernel_size=config.kernel_size)

        if use_cuda:
            model = nn.DataParallel(model).cuda()
        #model.cuda()

        optimizer = optim.Adam(params=model.parameters(), lr=config.lr, weight_decay=0.0)
        if hasattr(config.data, 'modelPath'):
            modelPath = np.array2string(np.squeeze(config.data.modelPath))[1:-1]
            print(modelPath)
            checkpoint = torch.load(modelPath)
            # print(checkpoint['model_state_dict'])
            model.load_state_dict(checkpoint['model_state_dict'])
            # print(checkpoint['optimizer_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # model = load_model_from(modelPath, use_cuda=True)
        #model = torch.load('snapshots/some_model')

        print('Model: ', model)
        print('Parameter count: ', model.module.parameter_count())

    if config.dataset=='librispeech':
        inputLocation = np.array2string(np.squeeze(config.eC.inputPath))[1:-1]+'speech/LibriSpeech/'
        dataset_name = 'dev-clean'
        dataset_name_eval = 'test-clean'
    else:
        inputLocation = np.array2string(np.squeeze(config.eC.inputPath))[1:-1]+'music/gtzan/'
        dataset_name = 'dev'
        dataset_name_eval = 'test'
    print(inputLocation)
    if config.step=='features':
        dataLocation = np.array2string(np.squeeze(config.eC.dataPath))[1:-1]
        dataLocation += 'features/'
        dataLocation += np.array2string(np.squeeze(config.eS.infoHash))[1:-1]
        dataLocationTrain = dataLocation+'_train'
        dataLocationTest = dataLocation+'_test'
    else:
        dataLocationTrain = np.array2string(np.squeeze(config.data.trainPath))[1:-1]
        dataLocationTest = np.array2string(np.squeeze(config.data.testPath))[1:-1]

    data = CNNDataset(dataset_file=dataLocationTrain,
                      file_location=inputLocation+dataset_name,
                      sampling_rate=5000,
                      block_size = config.sampling_rate,
                      normalize=True, compute=config.step=='features', squeeze=config.squeeze)

    data_eval = CNNDataset(dataset_file=dataLocationTest,
                      file_location=inputLocation+dataset_name_eval,
                           sampling_rate=config.sampling_rate,
                           block_size = config.block_size,
                           normalize=True, compute=config.step=='features', squeeze=config.squeeze)
    print('Dataset smat = hdf5storage.loadmatize:    ', len(data))

    if config.step!='features':
        trainer = CNNTrainer(model=model,
                             lr=config.lr,
                             weight_decay=0.0,
                             optimizer=optimizer,
                             snapshot_path=config.expLanes,
                             snapshot_interval=config.snapshot_interval,
                             dtype=dtype)

    if config.step=='train':
        print('----- Training -----')
        store, obs = trainer.train(dataset=data,
                      dataset_validation=data_eval,
                      batch_size=config.batch_size,
                      epochs=config.epochs,
                      target=config.target,
                      q = config.q)

    if config.step=='test':
        print('----- Evaluation -----')
        store, obs = trainer.test(dataset=data_eval, batch_size=config.block_size, save=True)

    if config.expLanes :
        if config.step=='features':
            store = {}
            obs = {}
            store['trainPath'] = data.dataset_file
            store['testPath'] = data_eval.dataset_file
            store['trainFiles'] = data.get_files()
            store['testFiles'] = data_eval.get_files()

            obs['nbBlocksTrain'] = len(data)
            obs['nbBlocksTest'] = len(data_eval)
        if config.step=='train':
            print('train')
        if config.step=='test':
            print('test')

        if os.path.exists(config.expLanes+'_data.mat'): os.remove(config.expLanes+'_data.mat')
        hdf5storage.savemat(config.expLanes+'_data', store)
        if os.path.exists(config.expLanes+'_obs.mat'):os.remove(config.expLanes+'_obs.mat')
        hdf5storage.savemat(config.expLanes+'_obs', obs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Datah
    parser.add_argument('--expLanes', type=str, default='')
    parser.add_argument('--dataset', type=str, default='dev-clean')
    parser.add_argument('--dataset_eval', type=str, default='test-clean')
    parser.add_argument('-load_mdl', action='store_true')

    # Logging
    parser.add_argument('--snapshot_interval', type=int, default=1000)
    parser.add_argument('--validation_interval', type=int, default=2000000)

    # Training
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)

    # Experience factors    if config.dataset_eval is not None:

    parser.add_argument('--target', type=str, default='spec', help='spec, wspec, cqt')
    parser.add_argument('--q', type=int, default=27)

    config = parser.parse_args()

    if config.expLanes :
        ec = hdf5storage.loadmat(config.expLanes)
        eSetting = ec['data']['info']['setting']
        eConfig = ec['data']['info']['config']
        # print(ec)

        eC = eConfig.view(np.recarray)
        eS = eSetting.view(np.recarray)

        config.step = np.squeeze(eC.stepName)[int(np.squeeze(eC.step.id).item(0)-1)]
        config.eC = eC
        config.eS = eS
        config.data = ec['data'].view(np.recarray)

        # print(sf)
        config.batch_size = int(np.nan_to_num(np.squeeze(eSetting['batchSize'])))
        config.block_size = int(np.nan_to_num(np.squeeze(eSetting['blockSize'])))
        config.sampling_rate = int(np.nan_to_num(np.squeeze(eSetting['samplingFrequency'])))
        config.squeeze = eSetting['squeeze']
        config.dataset = np.array2string(np.squeeze(eSetting['dataset']))[1:-1]
        if config.step!='features':
            config.kernel_size = int(np.nan_to_num(np.squeeze(eSetting['kernel_size'])))
            config.lr = float(np.squeeze(eSetting['learningRate']))
            config.epochs = int(np.nan_to_num(np.squeeze(eSetting['epochs'])))

        #print(config.epochs)
        main(config)
