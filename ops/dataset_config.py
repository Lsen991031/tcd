# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import os

ROOT_DATASET = '/media/ustc/30d7c9f1-d4ab-497a-b5c2-22e77be472c1/user/ls/mmaction2-master/data/'

def return_ucf101(modality):
    filename_categories = 'ucf101/annotations/classInd.txt'
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'ucf101/rawframes'
        filename_imglist_train = 'ucf101/ucf101_train_split_1_rawframes.txt'
        filename_imglist_val = 'ucf101/ucf101_val_split_1_rawframes.txt'
        prefix = 'img_{:05d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'ucf101/rawframes'
        filename_imglist_train = 'ucf101/splits/ucf101_flow_train_split_1.txt'
        filename_imglist_val = 'ucf101/splits/ucf101_flow_val_split_1.txt'
        prefix = 'flow_{}_{:05d}.jpg'
    elif modality == 'Both':
        root_data = ROOT_DATASET + 'ucf101/rawframes'
        filename_imglist_train = ['ucf101/ucf101_train_split_1_rawframes.txt', 'ucf101/splits/ucf101_flow_train_split_1.txt']
        filename_imglist_val = ['ucf101/ucf101_val_split_1_rawframes.txt', 'ucf101/splits/ucf101_flow_val_split_1.txt']
        prefix = ['img_{:05d}.jpg', 'flow_{}_{:05d}.jpg']
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_hmdb51(modality):
    filename_categories = 51
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'hmdb51/frames'
        filename_imglist_train = 'hmdb51/splits/hmdb51_rgb_train_split_1.txt'
        filename_imglist_val = 'hmdb51/splits/hmdb51_rgb_val_split_1.txt'
        prefix = 'img_{:05d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'hmdb51/frames'
        filename_imglist_train = 'hmdb51/splits/hmdb51_flow_train_split_1.txt'
        filename_imglist_val = 'hmdb51/splits/hmdb51_flow_val_split_1.txt'
        prefix = 'flow_{}_{:05d}.jpg'
    elif modality =='Both':
        root_data = ROOT_DATASET + 'hmdb51/frames'
        filename_imglist_train = ['hmdb51/splits/hmdb51_rgb_train_split_1.txt', 'hmdb51/splits/hmdb51_flow_train_split_1.txt']
        filename_imglist_val = ['hmdb51/splits/hmdb51_rgb_val_split_1.txt', 'hmdb51/splits/hmdb51_flow_val_split_1.txt']
        prefix = ['img_{:05d}.jpg', 'flow_{}_{:05d}.jpg']
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_something(modality):
    filename_categories = 'something/v1/category.txt'
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'something/v1/20bn-something-something-v1'
        filename_imglist_train = 'something/v1/train_videofolder.txt'
        filename_imglist_val = 'something/v1/val_videofolder.txt'
        prefix = '{:05d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'something/v1/20bn-something-something-v1-flow'
        filename_imglist_train = 'something/v1/train_videofolder_flow.txt'
        filename_imglist_val = 'something/v1/val_videofolder_flow.txt'
        prefix = 'flow_{}_{:05d}.jpg'
        #prefix = '{:06d}-{}_{:05d}.jpg'
    else:
        print('no such modality:'+modality)
        raise NotImplementedError
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_somethingv2(modality):
    filename_categories = 'sthsth2/splits/category.txt'
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'sthsth2/frames'
        filename_imglist_train = 'sthsth2/splits/train_videofolder.txt'
        filename_imglist_val = 'sthsth2/splits/val_videofolder.txt'
        prefix = 'img_{:05d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'sthsth2/flow'
        filename_imglist_train = 'sthsth2/splits/train_videofolder_flow.txt'
        filename_imglist_val = 'sthsth2/splits/val_videofolder_flow.txt'
        #prefix = '{:05d}.jpg'
        prefix = 'flow_{}_{:05d}.jpg'
    elif modality == 'Both':
        root_data = ROOT_DATASET + 'sthsth2/frames'
        filename_imglist_train = ['sthsth2/splits/train_videofolder.txt', 'sthsth2/splits/train_videofolder_flow.txt']
        filename_imglist_val = ['sthsth2/splits/val_videofolder.txt', 'sthsth2/splits/val_videofolder_flow.txt']
        prefix = ['img_{:05d}.jpg', 'flow_{}_{:05d}.jpg']
    else:
        raise NotImplementedError('no such modality:'+modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_minik(modality):
    filename_categories = 'minik/splits/category.txt'
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'minik/frames'
        filename_imglist_train = 'minik/splits/train_videofolder.txt'
        filename_imglist_val = 'minik/splits/val_videofolder.txt'
        prefix = 'img_{:05d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'minik/flow'
        filename_imglist_train = 'minik/splits/train_videofolder_flow.txt'
        filename_imglist_val = 'minik/splits/val_videofolder_flow.txt'
        prefix = 'flow_{}_{:05d}.jpg'
    elif modality == 'Both':
        root_data = ROOT_DATASET + 'minik/frames'
        filename_imglist_train = ['minik/splits/train_videofolder.txt', 'minik/splits/train_videofolder_flow.txt']
        filename_imglist_val = ['minik/splits/val_videofolder.txt', 'minik/splits/val_videofolder_flow.txt']
        prefix = ['img_{:05d}.jpg', 'flow_{}_{:05d}.jpg']
    else:
        raise NotImplementedError('no such modality:'+modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_jester(modality):
    filename_categories = 'jester/category.txt'
    if modality == 'RGB':
        prefix = '{:05d}.jpg'
        root_data = ROOT_DATASET + 'jester/20bn-jester-v1'
        filename_imglist_train = 'jester/train_videofolder.txt'
        filename_imglist_val = 'jester/val_videofolder.txt'
    else:
        raise NotImplementedError('no such modality:'+modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_kinetics(modality):
    filename_categories = 400
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'kinetics/images'
        filename_imglist_train = 'kinetics/labels/train_videofolder.txt'
        filename_imglist_val = 'kinetics/labels/val_videofolder.txt'
        prefix = 'img_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_dataset(dataset, modality):
    dict_single = {'jester': return_jester, 'something': return_something, 'somethingv2': return_somethingv2,
                   'ucf101': return_ucf101, 'hmdb51': return_hmdb51,
                   'kinetics': return_kinetics, 'minik':return_minik}
    if dataset in dict_single:
        file_categories, file_imglist_train, file_imglist_val, root_data, prefix = dict_single[dataset](modality)
    else:
        raise ValueError('Unknown dataset '+dataset)

    if modality == 'Both':
        file_imglist_train_rgb = os.path.join(ROOT_DATASET, file_imglist_train[0])
        file_imglist_val_rgb = os.path.join(ROOT_DATASET, file_imglist_val[0])
        file_imglist_train_flow = os.path.join(ROOT_DATASET, file_imglist_train[1])
        file_imglist_val_flow = os.path.join(ROOT_DATASET, file_imglist_val[1])
        file_imglist_train = [file_imglist_train_rgb, file_imglist_train_flow]
        file_imglist_val = [file_imglist_val_rgb, file_imglist_val_flow]
    else:
        file_imglist_train = os.path.join(ROOT_DATASET, file_imglist_train)
        file_imglist_val = os.path.join(ROOT_DATASET, file_imglist_val)
    if isinstance(file_categories, str):
        file_categories = os.path.join(ROOT_DATASET, file_categories)
        with open(file_categories) as f:
            lines = f.readlines()
        categories = [item.rstrip() for item in lines]
    else:  # number of categories
        categories = [None] * file_categories
    n_class = len(categories)
    print('{}: {} classes'.format(dataset, n_class))
    return n_class, file_imglist_train, file_imglist_val, root_data, prefix
