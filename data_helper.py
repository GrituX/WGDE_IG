import glob
import os
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pympi
import pandas as pd
import seaborn as sns

import torch
from torch.nn.functional import one_hot
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import OneHotEncoder

from logger import Logger


class VernissageDataset(Dataset):
    """
    Torch Dataset class for the training.
    """
    def __init__(self, data, labels):
        """
        :param data: Data array of shape (batch_size, sequence_length, number_features)
        :param labels: Label array, containing Mistrusting(0), Neutral(1), and Trusting(2)
        """
        if isinstance(data, np.ndarray):
            self.data = torch.from_numpy(data).float()
            self.labels = torch.from_numpy(labels).float()
        else:
            self.data = data.float()
            self.labels = labels.float()

    def __getitem__(self, index):
        if isinstance(index, list):
            assert len(index) < self.__len__(), 'Error: index out of range.'
        else:
            assert index < self.__len__(), 'Error: index out of range.'
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


class DataHelper:
    """
    Class to handle data loading and handling during training and prediction.
    """

    def __init__(self):
        self.interactions = [9, 10, 12, 15, 18, 19, 24, 26, 27, 30]
        self.annotations_dir = './Annotations'
        self.annotations_path = './Annotations/interaction_{}_segment.npy'
        self.feature_path = './Features/Aggregation/interaction_{}_features_step_*.npy'
        self.semantics_path = './Features/Semantics/Interaction_{}/semantics_reduced_segment_*.npy'
        self.audio_features_path = './Features/MFCC/interaction_{}_features_step_*.npy'
        self.files = sorted(glob.glob(self.annotations_dir + '/*.eaf'), key=len)

        self.label_dict = {'Mistrusting': 0, 'Neutral': 1, 'Trusting': 2}
        self.max_segment_len = 5000

        self.speech_categories = np.array(['silence', 'laughter', 'speech'])
        self.VFOA_categories = np.array(['PaintingL', 'PaintingC', 'PaintingR',
                                         'Nao', 'Participant', 'Other', 'Unclear'])
        self.vfoa_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        self.vfoa_encoder.fit(self.VFOA_categories.reshape(len(self.VFOA_categories), 1))

        self.feature_position = [('VFOA_p1', (0, 6)), ('VFOA_p2', (7, 13)), ('VFOA_p1_changes', (14, 14)),
                                 ('VFOA_p2_changes', (15, 15)), ('Len_ingroup_look', (16, 16)),
                                 ('Total_nod_time', (17, 17)), ('Vocal_p1', (18, 20)), ('Vocal_p2', (21, 23)),
                                 ('Vocal_Nao', (24, 24)), ('Speech_overlap', (25, 26)), ('FAU_p1', (27, 52)),
                                 ('FAU_p2', (53, 78)), ('Bar_x', (79, 80)), ('Bar_y', (81, 82)), ('CI_p1', (83, 84)),
                                 ('CI_p2', (85, 86)), ('Prosody_p1', (87, 96)), ('Prosody_p2', (97, 106)),
                                 ('Prosody_Nao', (107, 116)), ('Semantics_Nao', (117, 166)), ('MFCC_p1', (167, 174)),
                                 ('F0_d_p1', (175, 176)), ('MFCC_d_p1', (177, 184)), ('MFCC_p2', (185, 192)),
                                 ('F0_d_p2', (193, 194)), ('MFCC_d_p2', (195, 202)), ('MFCC_Nao', (203, 210)),
                                 ('F0_d_Nao', (211, 212)), ('MFCC_d_Nao', (213, 220))]

        self.continuous_cols = ['VFOA_p1_changes', 'VFOA_p2_changes', 'Len_ingroup_look', 'Total_nod_time',
                                'Speech_overlap', 'FAU_p1', 'FAU_p2', 'Bar_x', 'Bar_y', 'CI_p1', 'CI_p2', 'Prosody_p1',
                                'Prosody_p2', 'Prosody_Nao', 'Semantics_Nao', 'MFCC_p1', 'F0_d_p1', 'MFCC_d_p1',
                                'MFCC_p2', 'F0_d_p2', 'MFCC_d_p2', 'MFCC_Nao', 'F0_d_Nao', 'MFCC_d_Nao']

        self.robot_cols = ['Vocal_Nao', 'Prosody_Nao', 'Semantics_Nao', 'MFCC_Nao', 'F0_d_Nao', 'MFCC_d_Nao']

        self.human_cols = ['VFOA_p1', 'VFOA_p2', 'VFOA_p1_changes', 'VFOA_p2_changes', 'Len_ingroup_look',
                           'Total_nod_time', 'Vocal_p1', 'Vocal_p2', 'Speech_overlap', 'FAU_p1', 'FAU_p2', 'Bar_x',
                           'Bar_y', 'CI_p1', 'CI_p2', 'Prosody_p1', 'Prosody_p2', 'MFCC_p1', 'MFCC_p2', 'F0_d_p1',
                           'F0_d_p2', 'MFCC_d_p1', 'MFCC_d_p2']

        self.participant_cols = ['VFOA_p1', 'VFOA_p1_changes', 'Len_ingroup_look', 'Total_nod_time', 'Vocal_p1',
                                 'Speech_overlap', 'FAU_p1', 'Bar_x', 'Bar_y', 'CI_p1', 'Prosody_p1', 'MFCC_p1',
                                 'F0_d_p1', 'MFCC_d_p1']

        self.triad_cols = ['Bar_x', 'Bar_y']
        self.dyad_cols = ['Len_ingroup_look', 'Total_nod_time', 'Speech_overlap']

        _, (_, nb_features) = self.feature_position[-1]
        self.nb_features = nb_features + 1

        self.nb_modalities = 4

    def features2dataframe(self):
        _, (_, feature_length) = self.feature_position[-1]

        col_names = ['Interaction', 'Step', 'Label']
        feature_index = len(col_names)
        for i in range(len(self.feature_position)):
            col, (start, end) = self.feature_position[i]
            for l in range(end-start+1):
                col_names.append(col+'_{}'.format(l))

        df = pd.DataFrame(columns=col_names)

        for interaction in self.interactions:
            df_tmp = pd.DataFrame(columns=col_names)
            x_train, y_train, x_val, y_val, labels = self.get_data(leave='None', seq_length=1,
                                                                   which_out=[interaction], nb_out=1)
            assert x_val.shape[-1] == feature_length + 1  # Add 1 since the dict cols stores indexes.

            for i in range(feature_index, len(col_names)):
                df_tmp[col_names[i]] = x_val[:, 0, i-feature_index]
            df_tmp['Label'] = y_val
            df_tmp['Step'] = [i for i in range(len(df_tmp))]
            df_tmp['Interaction'] = interaction
            df = pd.concat([df, df_tmp])

        return df

    def get_feature_indices(self, feature_name=''):
        for name, (start, end) in self.feature_position:
            if feature_name == name:
                return [i for i in range(start, end+1)]
        return []

    def get_indices(self, what=None):
        col_indices = []
        cols_to_check = None
        if what == 'robot':
            cols_to_check = self.robot_cols
        elif what == 'human':
            cols_to_check = self.human_cols
        elif what == 'participant' or what == 'p1':
            cols_to_check = self.participant_cols
        elif what == 'p2':
            cols_to_check = self.participant_cols
            cols_to_check = [col.replace('1', '2') for col in cols_to_check]
        elif what == 'continuous':
            cols_to_check = self.continuous_cols
        elif what == 'group':
            cols_to_check = self.dyad_cols + self.triad_cols
        else:
            return None
        for name, (start, end) in self.feature_position:
            if any(elem in name for elem in cols_to_check):
                col_indices += [i for i in range(start, end+1)]
        return col_indices

    def get_labels(self):
        return list(self.label_dict.keys())

    def get_interaction_list(self):
        return self.interactions

    def get_feature_length(self, what='robot'):
        nb_features = 0
        cols_to_check = None
        if what == 'robot':
            cols_to_check = self.robot_cols
        elif what == 'human':
            cols_to_check = self.human_cols
        elif what == 'participant':
            cols_to_check = self.participant_cols
        for name, (start, end) in self.feature_position:
            if any(elem in name for elem in cols_to_check):
                nb_features += end - start + 1
        return nb_features

    def get_data_slice(self, x, what=None):
        return x[:, self.get_indices(what)]

    def remove_data(self, x, what=None):
        if type(what) == list:
            cols_to_remove = []
            for elem in what:
                cols_to_remove += self.get_indices(what)
        else:
            cols_to_remove = self.get_indices(what)
        return x[:, [i for i in range(self.nb_features) if i not in cols_to_remove]]

    def swap_participants(self, x, y):
        cols_p1 = self.participant_cols
        cols_p2 = [col.replace('1', '2') for col in cols_p1]
        assert len(cols_p1) == len(cols_p2)
        indices = []
        for i in range(len(cols_p1)):
            indices += self.get_feature_indices(cols_p2[i]) + self.get_feature_indices(cols_p1[i])
        out_x = None
        out_y = None
        if type(x) == np.ndarray:
            out_x = np.append(x, x[:, :, indices], axis=0)
            out_y = np.append(y, y, axis=0)
        elif type(x) == list:
            out_x = x.copy()
            out_y = y.copy()
            for i in range(len(x)):
                out_x.append(x[i][:, indices])
            out_y += y

        return out_x, out_y

    def split_modalities(self, x):
        return [np.append(x[:, 0:18], x[:, 27:79], axis=1), np.copy(x[:, 79:87]),
                np.append(x[:, 18:27], np.append(x[:, 87:117], x[:, 167:], axis=1), axis=1), x[:, 117:167]]

    def get_nb_modalities(self):
        return self.nb_modalities

    def retrieve_annotation_stats(self):
        annotations = []
        files = glob.glob(self.annotations_dir + '/*.eaf')
        # files = glob.glob('D:/Vernissage/Annotations_Raffaella/annotations/*.eaf')
        for file in files:
            eaf = pympi.Elan.Eaf(file)
            annots = sorted(eaf.get_annotation_data_for_tier('Trust'))
            annotations += annots
        annotations = np.array(annotations)
        labels = set(annotations[:, 2])
        df = pd.DataFrame(columns=['Label', 'Length'])
        print('-'*20)
        print('Before subsegmentation:')
        for label in labels:
            print('-'*10)
            print('Infos about label {}'.format(label))
            label_segment = annotations[annotations[:, 2] == label]
            lengths = (label_segment[:, 1].astype(int) - label_segment[:, 0].astype(int)).flatten()
            print('Count: {}\nMin: {}\t\tMax: {}\nMean: {:.2f}\t\tStd: {:.2f}'.format(len(lengths), lengths.min(),
                                                                                      lengths.max(), lengths.mean(),
                                                                                      lengths.std()))
        for label in labels:
            print('-'*20)
            print('Infos about label {}'.format(label))
            current = annotations[annotations[:, 2] == label]
            lengths = (current[:, 1].astype(int) - current[:, 0].astype(int)).flatten()
            split_segment = []
            for i in range(len(lengths)):
                if lengths[i] > 5000:
                    for j in range(lengths[i] // 5000):
                        split_segment.append(5000)
                    split_segment.append(lengths[i] % 5000)
            lengths = np.append(lengths, np.array(split_segment))
            lengths = lengths[(lengths > 600) & (lengths <= 5000)]
            sub_df = pd.DataFrame(columns=df.columns)
            sub_df['Length'] = lengths
            sub_df['Label'] = label
            df = pd.concat([df, sub_df])
            print('Count: {}\nMin: {}\t\tMax: {}\nMean: {:.2f}\t\tStd: {:.2f}'.format(len(lengths), lengths.min(),
                                                                                      lengths.max(), lengths.mean(),
                                                                                      lengths.std()))
        sns.histplot(data=df, hue='Label', x='Length', multiple='dodge', binwidth=200)
        plt.show()

    def interaction_combinations(self, n: int):
        for subset in itertools.combinations(self.interactions, n):
            yield list(subset)

    def retrieve_feature_files(self, file_format: str):
        files_to_search = file_format.replace('*', '{}')
        step = 0
        while os.path.exists(files_to_search.format(step)):
            step += 1
        return [files_to_search.format(i) for i in range(step)]

    def get_data(self, seq_length=10, leave='None', nb_out=1, which_out=[], y_as_seq=False):
        """
        :param modality: Modality to be loaded.
        :param seq_length: number of segments in a sequence. If == 0, then whole interaction is taken.
        :param leave: Get rid of which labels ?
        :param nb_out: Number of interactions for the test set.
        :param which_out: Interaction to be used for the test set.
        :param y_as_seq: Whether to return Y as a multi-output for each sequence.
        :return Data split in train and test.
        """

        tmp_x_train = []
        tmp_y_train = []
        tmp_x_test = []
        tmp_y_test = []

        if nb_out != 1 and len(which_out) != nb_out:
            one_out = np.random.choice(self.interactions, replace=False, size=(nb_out,))
        else:
            one_out = which_out
        Logger.log('Interaction {} is out.'.format(one_out))

        Logger.log('Reading annotation files...')
        # Get annotation files.
        for interaction in self.interactions:
            is_test = interaction in one_out
            file = self.annotations_path.format(interaction)
            tmp_annot = np.load(file)[10-seq_length:]
            if y_as_seq:
                tmp_annot = [np.array(tmp_annot[l:l+seq_length]).astype(str) for l in range(len(tmp_annot)-seq_length+1)]
            else:
                tmp_annot = tmp_annot[seq_length-1:]

            if is_test:
                tmp_y_test.append(tmp_annot)
            else:
                tmp_y_train.append(tmp_annot)

            feature_files = self.retrieve_feature_files(self.feature_path.format(interaction))
            semantics_files = self.retrieve_feature_files(self.semantics_path.format(interaction))
            added_audio_files = self.retrieve_feature_files(self.audio_features_path.format(interaction))
            assert len(feature_files) == len(semantics_files)
            assert len(feature_files) == len(added_audio_files)

            tmp_feat = [np.concatenate((np.load(feature_files[i]),
                                   np.load(semantics_files[i]),
                                   np.load(added_audio_files[i])))
                        for i in range(10-seq_length, len(feature_files))]

            length = len(tmp_feat) if seq_length == 0 else seq_length
            nb_iterations = int(len(tmp_feat) - length + 1)
            feature_seq = [np.array(tmp_feat[i:i+length]) for i in range(nb_iterations)]
            if is_test:
                tmp_x_test.append(np.array(feature_seq))
            else:
                tmp_x_train.append(np.array(feature_seq))
            Logger.log('Done reading data from interaction {}.'.format(interaction))

        x_train = []
        y_train = []
        x_test = []
        y_test = []

        for i in range(len(tmp_x_train)):
            for seq in tmp_x_train[i]:
                x_train.append(seq)
            for seq in tmp_y_train[i]:
                y_train.append(seq)

        for i in range(len(tmp_x_test)):
            for seq in tmp_x_test[i]:
                x_test.append(seq)
            for seq in tmp_y_test[i]:
                y_test.append(seq)

        if seq_length == 0:
            for i in range(len(x_train)):
                x_train[i] = torch.FloatTensor(x_train[i])
                y_train[i] = torch.FloatTensor(y_train[i])
            for i in range(len(x_test)):
                x_test[i] = torch.FloatTensor(x_test[i])
                y_test[i] = torch.FloatTensor(y_test[i])
        else:
            x_train = np.array(x_train, dtype=np.float32)
            y_train = np.array(y_train)
            x_test = np.array(x_test, dtype=np.float32)
            y_test = np.array(y_test)

        if seq_length != 0:
            x_train, y_train, x_test, y_test, labels = self.drop_label(leave, x_train, y_train, x_test, y_test)
        else:
            labels = self.get_labels()

        return x_train, y_train, x_test, y_test, labels

    def drop_label(self, leave, x_train, y_train, x_test, y_test):
        assert len(x_train) == len(y_train)
        labels = list(self.label_dict.keys())
        if leave == 'Neutral':
            indices = np.where(y_train == self.label_dict['Neutral'])
            y_train = np.delete(y_train, indices)
            x_train = np.delete(x_train, indices, axis=0)
            y_train[y_train == self.label_dict['Trusting']] = 1
            indices = np.where(y_test == self.label_dict['Neutral'])
            y_test = np.delete(y_test, indices)
            x_test = np.delete(x_test, indices, axis=0)
            y_test[y_test == self.label_dict['Trusting']] = 1
            labels = ['Mistrusting', 'Trusting']
        elif leave == 'Mistrusting':
            indices = np.where(y_train == self.label_dict['Mistrusting'])
            y_train = np.delete(y_train, indices)
            x_train = np.delete(x_train, indices, axis=0)
            y_train[y_train == self.label_dict['Neutral']] = 0
            y_train[y_train == self.label_dict['Trusting']] = 1
            indices = np.where(y_test == self.label_dict['Mistrusting'])
            y_test = np.delete(y_test, indices)
            x_test = np.delete(x_test, indices, axis=0)
            y_test[y_test == self.label_dict['Neutral']] = 0
            y_test[y_test == self.label_dict['Trusting']] = 1
            labels = ['Neutral', 'Trusting']
        elif leave == 'Trusting':
            indices = np.where(y_train == self.label_dict['Trusting'])
            y_train = np.delete(y_train, indices)
            x_train = np.delete(x_train, indices, axis=0)
            indices = np.where(y_test == self.label_dict['Trusting'])
            y_test = np.delete(y_test, indices)
            x_test = np.delete(x_test, indices, axis=0)
            labels = ['Mistrusting', 'Neutral']
        return x_train, y_train, x_test, y_test, labels

    def one_hot_encode(self, y_train, y_test, nb_classes):
        train_hot = None
        test_hot = None
        if type(y_train) == np.ndarray:
            idx = np.identity(nb_classes)
            train_hot = np.zeros((*y_train.shape, nb_classes))
            test_hot = np.zeros((*y_test.shape, nb_classes))
            for i in range(nb_classes):
                train_hot[y_train == i] = idx[i]
                test_hot[y_test == i] = idx[i]
        elif type(y_train) == list:
            train_hot = []
            test_hot = []
            for tensor in y_train:
                train_hot.append(one_hot(tensor.unsqueeze(0).to(torch.int64), num_classes=nb_classes))
            for tensor in y_test:
                test_hot.append(one_hot(tensor.unsqueeze(0).to(torch.int64), num_classes=nb_classes))
        return train_hot, test_hot


if __name__ == '__main__':

    helper = DataHelper()
    df_out = helper.features2dataframe()
    df_out.to_csv('./Features/feature_dataframe.csv', sep=',', index=False)
