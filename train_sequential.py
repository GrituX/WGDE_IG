import os.path
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional import f1_score, precision_recall
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score

from data_helper import VernissageDataset, DataHelper
from wgde_ig_modules import WGDEIG
from logger import Logger
import utility


def train_model_binary(model, train_dataloader, validation_dataloader, criterion, optimizer, interaction, epochs=50,
                       save_learning_curves=False, device=torch.device('cpu'), nb_classes=3, res_dir='./output_rnn/',
                       validate_every=1, summary_writer=None):
    """
    :param model: Model to train.
    :param train_dataloader: Loads data to train.
    :param validation_dataloader: Loads validation data to monitor training and save best model.
    :param criterion: Loss function used for training.
    :param optimizer: Optimizer used for training.
    :param interaction: Which interaction is used for validation.
    :param epochs: Number of epochs during which the model is trained.
    :param save_learning_curves: Should training and validation curves be saved.
    :param device: CPU or GPU.
    :param nb_classes: Number of classes.
    :param res_dir: Ouput directory where both the model and learning curves will be saved.
    :param validate_every: Evaluate the performance of the model on the validation set every validate_every epochs.
    :param summary_writer:
    :return: No return.
    """

    epoch = 0
    best_val_score = 0.
    best_epoch = 0

    if save_learning_curves:
        train_losses = []
        validation_losses = []

    Logger.log('Start training {}...'.format(model.name))

    model.zero_grad()
    while epoch <= epochs:
        model.train()
        Logger.log('Epoch {}/{}...'.format(epoch, epochs))

        if save_learning_curves:
            train_loss = 0

        for data, target in train_dataloader:
            optimizer.zero_grad()

            output = model(data.to(device).permute(1, 0, 2))
            target = target.view(output.size()).double().to(device)

            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            if save_learning_curves:
                train_loss += loss.item() * data.size(0)

        if save_learning_curves:
            Logger.log('Training loss: {}'.format(train_loss / len(train_dataloader.dataset)))
            summary_writer.add_scalar('Loss/train', train_loss / len(train_dataloader.dataset), epoch)
            train_losses.append(train_loss / len(train_dataloader.dataset))

        if epoch % validate_every == 0:
            with torch.no_grad():
                model.eval()
                validation_loss = 0
                val_output = np.array([])
                val_target = np.array([])

                for validation_data, validation_target in validation_dataloader:
                    validation_output = model(validation_data.to(device).permute(1, 0, 2))
                    validation_target = validation_target.view(validation_output.size()).double().to(device)
                    val_output = np.append(val_output, validation_output.detach().softmax(dim=1).cpu().numpy())
                    val_target = np.append(val_target, validation_target.cpu().numpy())

                    if save_learning_curves:
                        validation_loss += criterion(validation_output.detach(),
                                                     validation_target).item() * validation_data.size(0)

                if save_learning_curves:
                    validation_losses.append(validation_loss / len(validation_dataloader.dataset))
                    summary_writer.add_scalar('Loss/val', validation_loss / len(validation_dataloader.dataset), epoch)

                try:
                    val_score = roc_auc_score(val_target.reshape(-1, nb_classes),
                                              val_output.reshape(-1, nb_classes))
                except ValueError:
                    val_score = np.nan
                summary_writer.add_scalar('Accuracy/val', val_score, epoch)
                Logger.log('Validation score: {:.5f} / Best: {:.5f}'.format(val_score, best_val_score))

                if val_score > best_val_score:
                    best_val_score = val_score
                    best_epoch = epoch
                    Logger.log('Model saved at epoch {} for a score {}'.format(epoch, val_score))
        epoch += 1

    Logger.log('Training finished !')
    model.save(res_dir, epoch)

    val_output = val_output.reshape(-1, nb_classes)
    val_target = val_target.reshape(-1, nb_classes)
    try:
        f1 = f1_score(torch.tensor(val_output, dtype=torch.float),
                      torch.tensor(val_target, dtype=torch.int),
                      num_classes=nb_classes).item()
        precision = precision_score(val_target.argmax(axis=1), val_output.argmax(axis=1))
        recall = recall_score(val_target.argmax(axis=1), val_output.argmax(axis=1))
    except ValueError:
        f1 = np.nan
        precision = np.nan
        recall = np.nan
    utility.plot_confusion(target=val_target.argmax(axis=1), output=val_output.argmax(axis=1),
                           out_name=res_dir + 'confusion_int_{}_{}_epoch_{}.png'.format(
                               interaction, model.name, epoch))

    if save_learning_curves:
        np.save('{}train-losses_interaction_{}.npy'.format(res_dir, interaction), train_losses)
        np.save('{}val-losses_interaction_{}.npy'.format(res_dir, interaction), validation_losses)
    Logger.log('Learning curves saved at ' + res_dir)
    plt.close()
    return best_val_score, best_epoch, val_score, epoch, f1, precision, recall


def train_model_multiclass(model, train_dataloader, validation_dataloader, criterion, optimizer, interaction, epochs=50,
                           save_learning_curves=False, device=torch.device('cpu'), nb_classes=3,
                           res_dir='./output_rnn/', validate_every=1, summary_writer=None):
    """
    :param model: Model to train.
    :param train_dataloader: Loads data to train.
    :param validation_dataloader: Loads validation data to monitor training and save best model.
    :param criterion: Loss function used for training.
    :param optimizer: Optimizer used for training.
    :param interaction: Which interaction is used for validation.
    :param epochs: Number of epochs during which the model is trained.
    :param save_learning_curves: Should training and validation curves be saved.
    :param device: CPU or GPU.
    :param nb_classes: Number of classes.
    :param res_dir: Ouput directory where both the model and learning curves will be saved.
    :param validate_every: Evaluate the performance of the model on the validation set every validate_every epochs.
    :param summary_writer:
    :return: No return.
    """

    epoch = 0
    best_val_score = 0.
    best_epoch = 0

    if save_learning_curves:
        train_losses = []
        validation_losses = []

    Logger.log('Start training {}...'.format(model.name))

    model.zero_grad()
    while epoch <= epochs:
        model.train()
        Logger.log('Epoch {}/{}...'.format(epoch, epochs))

        if save_learning_curves:
            train_loss = 0

        for data, target in train_dataloader:
            optimizer.zero_grad()

            output = model(data.to(device).permute(1, 0, 2))
            target = target.long().to(device)
            output = output.softmax(dim=1)

            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            if save_learning_curves:
                train_loss += loss.item() * data.size(0)

        if save_learning_curves:
            Logger.log('Training loss: {}'.format(train_loss / len(train_dataloader.dataset)))
            summary_writer.add_scalar('Loss/train', train_loss / len(train_dataloader.dataset), epoch)
            train_losses.append(train_loss / len(train_dataloader.dataset))

        if epoch % validate_every == 0:
            with torch.no_grad():
                model.eval()
                validation_loss = 0
                val_output = np.array([])
                val_target = np.array([])

                for validation_data, validation_target in validation_dataloader:
                    validation_output = model(validation_data.to(device).permute(1, 0, 2))
                    validation_target = validation_target.long().to(device)
                    val_output = np.append(val_output, validation_output.detach().softmax(dim=1).cpu().numpy())
                    val_target = np.append(val_target, validation_target.cpu().numpy())

                    if save_learning_curves:
                        validation_loss += criterion(validation_output.detach(), validation_target).item() * validation_data.size(0)

                if save_learning_curves:
                    validation_losses.append(validation_loss / len(validation_dataloader.dataset))
                    summary_writer.add_scalar('Loss/val', validation_loss / len(validation_dataloader.dataset), epoch)

                val_output = val_output.reshape(-1, nb_classes)
                val_score = f1_score(torch.tensor(val_output, dtype=torch.float),
                                     torch.tensor(val_target, dtype=torch.int), num_classes=nb_classes).item()
                Logger.log('Validation score: {:.5f} / Best: {:.5f}'.format(val_score, best_val_score))

                if val_score > best_val_score:
                    best_val_score = val_score
                    best_epoch = epoch
                    Logger.log('Model saved at epoch {} for a score {}'.format(epoch, val_score))
        epoch += 1

    Logger.log('Training finished !')
    model.save(res_dir, epoch)

    val_output = val_output.reshape(-1, nb_classes)
    try:
        f1 = f1_score(torch.tensor(val_output, dtype=torch.float),
                      torch.tensor(val_target, dtype=torch.int),
                      num_classes=nb_classes).item()
        precision, recall = precision_recall(torch.tensor(val_output, dtype=torch.float),
                                             torch.tensor(val_target, dtype=torch.int),
                                             average='micro', num_classes=nb_classes)
        precision, recall = precision.item(), recall.item()
    except ValueError:
        f1 = np.nan
        precision = np.nan
        recall = np.nan
    utility.plot_confusion(target=val_target, output=val_output.argmax(axis=1),
                           out_name=res_dir + 'confusion_int_{}_{}_epoch_{}.png'.format(interaction, model.name, epoch))

    if save_learning_curves:
        np.save('{}train-losses_interaction_{}.npy'.format(res_dir, interaction), train_losses)
        np.save('{}val-losses_interaction_{}.npy'.format(res_dir, interaction), validation_losses)
    Logger.log('Learning curves saved at ' + res_dir)
    plt.close()
    return best_val_score, best_epoch, val_score, epoch, f1, precision, recall


def rnn_train(rnn_archi='SimpleRNN', hidden_dims=[16, 16], epochs=50, seq_length=10, leave='None', batch_size=64,
              nb_out=3, lr=1e-3, weight_decay=0., save_learning_curves=1, split_participants=0, use_attention=0,
              out_dir='./output_rnn/'):

    results_df = pd.DataFrame({}, columns=['Leave', 'Architecture', 'H_dim', 'Seq_len', 'Int_out',
                                           'Best_score', 'Best_epoch', 'Last_score', 'Last_epoch', 'F1', 'Precision',
                                           'Recall', 'LR', 'Batch_size', 'Seed'])

    helper = DataHelper()
    for interaction in helper.interaction_combinations(nb_out):
        x_train, y_train, x_val, y_val, labels = helper.get_data(leave=leave, seq_length=seq_length,
                                                                 which_out=interaction, nb_out=nb_out)

        nb_classes = len(labels)
        if nb_classes == 2:
            y_train, y_val = helper.one_hot_encode(y_train, y_val, nb_classes)

        Logger.log('Preparing everything for the training...')
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Rescale continuous columns
        continous_cols = helper.get_indices(what='continuous')
        tmp_x_train = x_train.reshape((-1, x_train.shape[-1]))
        scaler = StandardScaler().fit(tmp_x_train[:, continous_cols])
        x_t_scaled = x_train.copy()
        x_v_scaled = x_val.copy()
        x_t_scaled[:, :, continous_cols] = scaler.transform(tmp_x_train[:, continous_cols]).reshape(
            (-1, seq_length, len(continous_cols)))
        x_v_scaled[:, :, continous_cols] = scaler.transform(x_val.reshape(-1, x_val.shape[-1])[:, continous_cols])\
            .reshape((-1, seq_length, len(continous_cols)))

        # Add noise
        noisy_x = x_t_scaled.copy()
        noisy_y = y_train.copy()
        for i in range(4):
            noise = np.random.normal(scale=2e-3, size=noisy_x.shape)
            x_t_scaled = np.append(x_t_scaled, noisy_x + noise, axis=0)
            y_train = np.append(y_train, noisy_y, axis=0)

        # Weights for class imbalance
        counts = []
        for label in range(nb_classes):
            comparison = label if nb_classes > 2 else np.identity(nb_classes)[label]
            class_count = np.count_nonzero(y_train == comparison)
            Logger.log('Class {}, count {}'.format(labels[label], class_count))
            counts.append(class_count)
        weights = torch.tensor(counts) / sum(counts)
        if nb_classes == 2:
            weights = 1. - weights
        else:
            weights = 1. / weights
            weights = weights / weights.sum()
        Logger.log('Weights for the training: {}'.format(weights))

        x_t_scaled, y_train = torch.from_numpy(x_t_scaled).double(), torch.from_numpy(y_train).double()
        x_v_scaled, y_val = torch.from_numpy(x_v_scaled).double(), torch.from_numpy(y_val).double()

        train_dataset = VernissageDataset(x_t_scaled, y_train)
        val_dataset = VernissageDataset(x_v_scaled, y_val)

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        is_multi_output = seq_length == 0
        participant_in_dim = helper.get_feature_length(what='participant')
        robot_dim = helper.get_feature_length(what='robot')
        group_dim = helper.get_feature_length(what='human')
        group_only_dim = group_dim - 2 * participant_in_dim

        seeds = np.array(np.random.rand(5) * 100, dtype=int)
        for seed in seeds:
            tensor_out = './tensorboard_runs/{}-split_{}/'.format(split_participants, rnn_archi)
            if not os.path.exists(tensor_out):
                os.mkdir(tensor_out)
            tensor_out = tensor_out + 'seqlen_{}/'.format(seq_length)
            if not os.path.exists(tensor_out):
                os.mkdir(tensor_out)
            writer = SummaryWriter(tensor_out + '{}_seed_{}/'.format(interaction, seed))
            model = WGDEIG(data_helper=helper, robot_input_dim=robot_dim, group_input_dim=group_dim,
                           hidden_dims=hidden_dims, rnn_archi=rnn_archi, split_participants=split_participants,
                           p_rnn_hdim=8, participant_in_dim=participant_in_dim,
                           group_only_dim=group_only_dim, use_attention=use_attention, batch_size=batch_size,
                           seq_len=seq_length, nb_classes=nb_classes, is_multi_output=is_multi_output,
                           int_out=interaction, drop_robot=True, device=device).to(device)
            Logger.log('Total number of parameters for model: {}'.format(model.get_total_parameters()))

            if nb_classes == 2:
                criterion = nn.BCEWithLogitsLoss(pos_weight=weights).to(device)
            else:
                criterion = nn.CrossEntropyLoss(weight=weights.double()).to(device)

            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            output_dir = out_dir + 'seed_{}/'.format(seed)
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)

            torch.manual_seed(seed)
            if nb_classes == 2:
                best_val_score, best_epoch, last_val_score, last_epoch, f1, precision, recall = \
                    train_model_binary(model, train_dataloader, val_dataloader, criterion, optimizer, interaction,
                                       epochs, save_learning_curves, device, nb_classes, res_dir=output_dir,
                                       summary_writer=writer)
            else:
                best_val_score, best_epoch, last_val_score, last_epoch, f1, precision, recall = \
                    train_model_multiclass(model, train_dataloader, val_dataloader, criterion, optimizer, interaction,
                                           epochs, save_learning_curves, device, nb_classes, res_dir=output_dir,
                                           summary_writer=writer)

            results_df.loc[-1] = [leave, rnn_archi, hidden_dims, seq_length, interaction,
                                  best_val_score, best_epoch, last_val_score, last_epoch, f1, precision, recall,
                                  lr, batch_size, seed]
            results_df.index += 1

    results_df.to_csv(out_dir + 'rnn_score.csv', sep=',', index=False)
    Logger.log('Done !')


def main():
    """
    Launch the training for a sequential approach.
    There are currently four types of architecture implemented, depending on whether participant data is split,
    and on how we model the interaction between the human group and the robot.
    One of the important parameters is seq_length. If it's set to 0, then we take the entire interaction as input,
    and formulate the classification as a multi output problem.
    List of parameters:
    rnn_archi : Which RNN Architecture to use ? Options are 'Simple' for a GRU RNN or 'IG' to use the IG module.
    nb_out : Number of interactions that are out in the test set. The program automatically generates all possible pairs
    seq_len : Length of the sequence given as input of the model.
    leave : Which label to drop for the training ('Mistrusting', 'Neutral', 'Trusting', or 'None')
    hidden_dims : Dimensions of the hidden state in the IG module for the robot and the group.
    split_participants : Should the WGDE module be used ?
    use_attention : Whether to use attention or not
    epochs : Number of epochs to train the model for.
    batch_size : Batch size
    lr : Learning rate of the training.
    weight_decay : Value of the weight decay applied in the loss (L2 penalty)
    save_learning_curves : Save learning curves ?
    out_dir : Path to the directory where results will be saved.
    """
    parser = argparse.ArgumentParser(description='Train a model for automatic trust level recognition.')
    parser.add_argument('--rnn_archi', type=str, default='Simple')
    parser.add_argument('--nb_out', type=int, default=1)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--leave', type=str, default='None')
    parser.add_argument('--hidden_dim', nargs='+', type=int, default=[32, 32])
    parser.add_argument('--split_participants', type=int, default=0)
    parser.add_argument('--use_attention', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--save_learning_curves', type=int, default=1)
    parser.add_argument('--out_dir', type=str, default='./Results/')
    args = parser.parse_args()

    out_dir = args.out_dir + '{}_out/{}-split_{}/seqlen_{}/'.format(args.leave, args.split_participants,
                                                                    args.architecture, args.seq_len)
    utility.create_dir_for_path(out_dir)
    Logger.out_dir = out_dir

    rnn_train(rnn_archi=args.rnn_archi, hidden_dims=args.hidden_dim, epochs=args.epochs,
              leave=args.leave, batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
              save_learning_curves=args.save_learning_curves, seq_length=args.seq_len, out_dir=out_dir,
              split_participants=args.split_participants, use_attention=args.use_attention, nb_out=args.nb_out)


if __name__ == '__main__':
    main()
