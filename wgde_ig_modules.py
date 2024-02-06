import math
import torch
import torch.nn as nn
import numpy as np


class IGCell(nn.Module):
    def __init__(self, robot_input_dim, group_input_dim, r_hdim, g_hdim):
        """
        :param robot_input_dim: The input dimension of robot data.
        :param group_input_dim: The input dimension of user group data.
        :param r_hdim: Hidden dimension for the robot cell.
        :param g_hdim: Hidden dimension for the user group cell.
        """
        super(IGCell, self).__init__()
        self.robot_input_dim = robot_input_dim
        self.group_input_dim = group_input_dim
        self.r_hdim = r_hdim
        self.g_hdim = g_hdim

        self.robot_cell = nn.GRUCell(self.robot_input_dim + self.g_hdim, self.r_hdim)
        self.group_cell = nn.GRUCell(self.group_input_dim + self.r_hdim, self.g_hdim)

    def forward(self, data_r, data_u, r_prev, g_prev):
        """
        :param data_r: Robot data input.
        :param data_u: User group data input.
        :param r_prev: Hidden state of robot data from previous step.
        :param g_prev: Hidden state of user group data from previous step.
        :return Tuple consisting of Robot hidden state and User group hidden state.
        """
        r_hidden = self.robot_cell(torch.cat((data_r, g_prev), dim=1), r_prev)
        g_hidden = self.group_cell(torch.cat((data_u, r_hidden), dim=1), g_prev)
        return r_hidden, g_hidden


class WGDECell(nn.Module):
    def __init__(self, data_helper, group_input_dim, split_participants=0,
                 p_rnn_hdim=20, participant_in_dim=49, group_only_dim=0, batch_size=64, device='cpu'):
        """
        :param group_input_dim: Output dimension of the Within-Group Dynamics Encoder.
        :param split_participants: Should user group dynamics be encoded or left as is ?
        :param p_rnn_hdim: Hidden dimension for the cell of each user.
        :param participant_in_dim: Input dimension for data from each user.
        :param group_only_dim: Input dimension for dyadic and triadic data.
        :param batch_size: Batch size for training.
        :param device: Torch device to use for training.
        """
        super(WGDECell, self).__init__()
        self.group_input_dim = group_input_dim
        self.split_participants = split_participants
        self.group_only_dim = group_only_dim
        self.p_rnn_hdim = p_rnn_hdim
        self.batch_size = batch_size
        self.device = device
        self.data_helper = data_helper

        self.p1_hidden = torch.zeros(self.batch_size, self.p_rnn_hdim).double()
        self.p2_hidden = torch.zeros(self.batch_size, self.p_rnn_hdim).double()

        if self.split_participants:
            self.p1_cell = nn.GRUCell(input_size=participant_in_dim, hidden_size=p_rnn_hdim).double()
            self.p2_cell = nn.GRUCell(input_size=participant_in_dim, hidden_size=p_rnn_hdim).double()
            self.group_fc = nn.Sequential(
                nn.Linear(self.p_rnn_hdim * 2 + self.group_only_dim, 16),
                nn.ReLU()
            ).double()
            self.group_input_dim = 16

    def get_output_dim(self):
        """
        :return Output dimension of the Within-Group Dynamics Encoder.
        """
        return self.group_input_dim

    def init_hidden(self, batch_size):
        """
        :param batch_size: Batch size for training.
        """
        if self.split_participants:
            self.p1_hidden = torch.zeros(batch_size, self.p_rnn_hdim).double().to(self.device)
            self.p2_hidden = torch.zeros(batch_size, self.p_rnn_hdim).double().to(self.device)

    def forward(self, x):
        """
        :param x: The entire feature vector.
        :return If data is split: encoding of the group. If not: human data filtered from the entire feature vector.
        """
        if self.split_participants:
            x_p1 = self.data_helper.get_data_slice(x, 'p1').double()
            x_p2 = self.data_helper.get_data_slice(x, 'p2').double()
            x_group = self.data_helper.get_data_slice(x, 'group').double()

            self.p1_hidden = self.p1_cell(x_p1, self.p1_hidden)
            self.p2_hidden = self.p2_cell(x_p2, self.p2_hidden)
            xt_g = self.group_fc(torch.cat((self.p1_hidden, self.p2_hidden, x_group), dim=1))
            return xt_g
        else:
            return self.data_helper.get_data_slice(x, 'human')


class SelfAttention(nn.Module):
    def __init__(self, dim, device='cpu'):
        """
        :param dim: Dimension of the self attention vector.
        :param device: Torch device.
        """
        super(SelfAttention, self).__init__()
        self.dim = dim
        self.q = nn.Parameter(torch.randn(dim).to(device).double(), requires_grad=True)
        self.device = device

    def forward(self, x, mask=None):
        """
        :param x: Input data.
        :param mask: Mask to apply on the attention logits.
        :return Tuple of the attention values, and attention vector.
        """
        attn_logits = torch.matmul(x, self.q) / math.sqrt(self.dim)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
        attention = torch.softmax(attn_logits, dim=0)
        value = torch.mul(attention.unsqueeze(-1), x)
        return value, attention


class IGRNN(nn.Module):
    def __init__(self, data_helper, robot_input_dim, group_input_dim, r_hdim, g_hdim,
                 split_participants=0, p_rnn_hdim=10, participant_in_dim=49, group_only_dim=0,
                 batch_size=64, device='cpu'):
        """
        :param robot_input_dim: Input dimension of robot data.
        :param group_input_dim: Input dimension of user group data.
        :param r_hdim: Hidden dimension for the robot RNN in the Interactional GRU.
        :param g_hdim: Hidden dimension for the group RNN in the Interactional GRU.
        :param split_participants: Should the Within-Group Encoder be activated or not ?
        :param p_rnn_hdim: Hidden dimension for the user RNN in the Within-Group Dynamics Encoder.
        :param participant_in_dim: Input dimension for data from each user.
        :param group_only_dim: Input dimension for dyadic and triadic data.
        :param batch_size: Batch size for training.
        :param device: Torch device.
        """
        super(IGRNN, self).__init__()
        self.robot_input_dim = robot_input_dim
        self.group_input_dim = group_input_dim
        self.r_hdim = r_hdim
        self.g_hdim = g_hdim
        self.p_rnn_hdim = p_rnn_hdim
        self.group_only_dim = group_only_dim
        self.batch_size = batch_size
        self.device = device
        self.r_hidden = torch.zeros(self.batch_size, self.r_hdim).double()
        self.g_hidden = torch.zeros(self.batch_size, self.g_hdim).double()
        self.data_helper = data_helper

        self.wgde_cell = WGDECell(data_helper=self.data_helper, group_input_dim=group_input_dim,
                                  split_participants=split_participants, p_rnn_hdim=p_rnn_hdim,
                                  participant_in_dim=participant_in_dim, group_only_dim=group_only_dim,
                                  batch_size=self.batch_size, device=device).to(device)

        if split_participants:
            trust_group_input_dim = 16
        else:
            trust_group_input_dim = group_input_dim

        self.ig_cell = IGCell(robot_input_dim=self.robot_input_dim, group_input_dim=trust_group_input_dim,
                              r_hdim=self.r_hdim, g_hdim=self.g_hdim).double()

    def init_hidden(self, batch_size):
        """
        :param batch_size: Batch size for training.
        """
        self.r_hidden = torch.zeros(batch_size, self.r_hdim).double().to(self.device)
        self.g_hidden = torch.zeros(batch_size, self.g_hdim).double().to(self.device)

    def forward(self, sequence):
        """
        :param sequence: Sequence of time data, size (seq_len, batch, input_dim)
        :return: last group state in the sequence, size (batch, g_hdim).
        """

        self.init_hidden(sequence.size()[1])
        self.wgde_cell.init_hidden(sequence.size()[1])
        all_hidden = torch.zeros(0).double().to(self.device)

        for t, xt in enumerate(sequence):
            xt_r = self.data_helper.get_data_slice(xt, 'robot')
            xt_g = self.wgde_cell(xt)
            self.r_hidden, self.g_hidden = self.ig_cell(xt_r, xt_g, self.r_hidden, self.g_hidden)
            all_hidden = torch.cat((all_hidden, self.g_hidden.view(1, self.g_hidden.shape[0], -1)), dim=0)

        return all_hidden


class SimpleRNN(nn.Module):
    def __init__(self, data_helper, robot_input_dim, hidden_dim, split_participants=0, group_input_dim=106,
                 p_rnn_hdim=20, participant_in_dim=49, group_only_dim=0, batch_size=64, drop_robot=False, device='cpu'):
        """
        :param robot_input_dim: Input dimension of robot data.
        :param hidden_dim: Hidden dimension of the RNN.
        :param group_input_dim: Input dimension of user group data.
        :param split_participants: Should the Within-Group Encoder be activated or not ?
        :param p_rnn_hdim: Hidden dimension for the user RNN in the Within-Group Dynamics Encoder.
        :param participant_in_dim: Input dimension for data from each user.
        :param group_only_dim: Input dimension for dyadic and triadic data.
        :param batch_size: Batch size for training.
        :param drop_robot: Should robot data be dropped ?
        :param device: Torch device.
        """
        super(SimpleRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.drop_robot = drop_robot
        self.split_participants = split_participants
        self.device = device
        self.data_helper = data_helper

        self.group_cell = WGDECell(data_helper=self.data_helper, group_input_dim=group_input_dim,
                                   split_participants=split_participants, p_rnn_hdim=p_rnn_hdim,
                                   participant_in_dim=participant_in_dim, group_only_dim=group_only_dim,
                                   batch_size=batch_size, device=device).to(device)

        self.input_dim = self.group_cell.get_output_dim()
        if not self.drop_robot:
            self.input_dim += robot_input_dim

        self.rnn_cell = nn.GRUCell(input_size=self.input_dim, hidden_size=self.hidden_dim).double()

    def init_hidden(self, batch_size):
        """
        :param batch_size: Batch size for training.
        """
        return torch.zeros(batch_size, self.hidden_dim).double().to(self.device)

    def forward(self, sequence):
        """
        :param sequence: Sequence of time data, size (seq_len, batch, input_dim)
        :return: last group state in the sequence, size (batch, hidden_dim).
        """
        hid = self.init_hidden(sequence.size()[1])
        self.group_cell.init_hidden(sequence.size()[1])
        all_hidden = torch.zeros(0).double().to(self.device)
        for t, xt in enumerate(sequence):
            out = self.group_cell(xt.double())
            if not self.drop_robot:
                out = torch.cat((out, self.data_helper.get_data_slice(xt, 'robot')), dim=1)
            hid = self.rnn_cell(out, hid)
            all_hidden = torch.cat((all_hidden, hid.view(1, hid.shape[0], -1)), dim=0)
        return all_hidden


class MLP(nn.Module):
    def __init__(self, input_dim, nb_output):
        """
        :param input_dim: Input dimension of the Multi-layer perceptron.
        :param nb_output: Number of classes of the Multi-layer perceptron.
        """
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.nb_output = nb_output

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, nb_output)
        ).double()

    def forward(self, x):
        if len(x.shape) == 3:
            tmp_x = x.permute(1, 0, 2)
            return self.fc(tmp_x)
        else:
            return self.fc(x)


class WGDEIG(nn.Module):
    """
    WGDEIG (Within-Group Dynamics Encoder (WGDE) + Interactional GRU (IG)) is a module to perform classification
    of TURIN trust labels based on a sequence of features from participants, the robot, and the entire group.
    """
    def __init__(self, data_helper, robot_input_dim, group_input_dim, hidden_dims=None, rnn_archi='Simple',
                 split_participants=0, p_rnn_hdim=10, participant_in_dim=10, group_only_dim=0, nb_classes=2,
                 use_attention=0, batch_size=64, seq_len=10, is_multi_output=False, drop_robot=False,
                 int_out=[], device='cpu'):
        """
        :param robot_input_dim: Input dimension of the robot data.
        :param group_input_dim: Input dimension of the group data.
        :param hidden_dims: Array containing the dimensions of the hidden state in the IG module for the robot and the group
        :param rnn_archi: 'Simple' to use a simple GRU RNN or 'IG' to use the IG module after the WGDE module.
        :param split_participants: Boolean to use the WGDE module or not.
        :param p_rnn_hdim: Hidden dimension of the participant's RNN in the WGDE module.
        :param participant_in_dim: Input dimension of each participant's data in the WGDE module.
        :param group_only_dim: Input dimension of dyadic and triadic data in the WGDE module.
        :param nb_classes: Number of classes to predict.
        :param use_attention: Boolean to use attention after the IG module.
        :param batch_size: Batch size during training.
        :param seq_len: Length of the input sequence.
        :param is_multi_output: Boolean to perform a multiple output prediction on the entire interaction.
        :param drop_robot: Boolean to drop robot data from the input. Not taken into account when the IG module is used.
        :param int_out: Array containing the interactions' number that are in the test set (used in the name of the architecture when saved).
        :param device: Torch device to train the model on.
        """
        super(WGDEIG, self).__init__()
        self.rnn_archi = rnn_archi
        self.use_attention = use_attention
        self.split_participants = split_participants
        self.p_rnn_hdim = p_rnn_hdim
        self.hidden_dims = hidden_dims
        self.seq_len = seq_len
        self.is_multi_output = is_multi_output
        self.device = device
        self.att_weights = None
        self.data_helper = data_helper
        self._name = '{}_GE_{}_hidden_{}_prnn_{}_out_{}'.format(
            self.rnn_archi, self.split_participants, self.hidden_dims, self.p_rnn_hdim, int_out)

        if self.architecture == 'IGR':
            assert len(hidden_dims) == 2, 'Argument \'hidden_dims\' is not of length 2.'
            self.rnn_module = IGRNN(self.data_helper, robot_input_dim, group_input_dim, r_hdim=hidden_dims[0],
                                    g_hdim=hidden_dims[1], split_participants=split_participants,
                                    p_rnn_hdim=p_rnn_hdim, participant_in_dim=participant_in_dim,
                                    group_only_dim=group_only_dim, batch_size=batch_size, device=device).to(device)
            self.att_dim = hidden_dims[1]
        else:
            self.rnn_module = SimpleRNN(data_helper=self.data_helper, robot_input_dim=robot_input_dim,
                                        hidden_dim=hidden_dims[0], split_participants=split_participants,
                                        group_input_dim=group_input_dim, p_rnn_hdim=p_rnn_hdim,
                                        participant_in_dim=participant_in_dim, group_only_dim=group_only_dim,
                                        batch_size=batch_size, drop_robot=drop_robot, device=device).to(device)
            self.att_dim = hidden_dims[0]

        if self.use_attention:
            self.attention = SelfAttention(dim=self.att_dim, device=device).to(device).double()

        self.mlp = MLP(input_dim=hidden_dims[1], nb_output=nb_classes).to(device)
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, sequence):
        """
        :param sequence: Input sequence of size (sequence_length, batch_size, nb_features).
        """
        if self.is_multi_output:
            output = torch.zeros(0).to(self.device)
            length = sequence.shape[0]
            rnn_out = self.rnn_module(sequence)
            for seq_len in range(1, length + 1):
                xt = rnn_out[:seq_len]
                if self.use_attention:
                    att_out, att_out_weights = self.attention(x=xt)
                    mlp_in = att_out.sum(dim=0)
                    self.att_weights = att_out_weights
                else:
                    mlp_in = rnn_out[-1]
                tmp_out = self.mlp(mlp_in)
                output = torch.cat((output, tmp_out), dim=0)
        else:
            # Run the sequence through the WGDE and IG modules.
            rnn_out = self.rnn_module(sequence)
            if self.use_attention:
                # Compute the attention values.
                att_out, att_out_weights = self.attention(x=rnn_out)
                mlp_in = att_out.sum(dim=0)
                self.att_weights = att_out_weights
            else:
                # Take the last hidden state of the RNN as input for the Multi-Layer Perceptron
                mlp_in = rnn_out[-1]
            # Apply dropout to reduce over-fitting.
            mlp_in = self.dropout(mlp_in)
            output = self.mlp(mlp_in)
        # We do not use sigmoid here since we use a BCEWithLogitsLoss that integrates the sigmoid already.
        return output

    def save(self, output, epoch):
        torch.save(self.state_dict(), output + self._name + '_epoch_{}.pt'.format(epoch))
        if self.use_attention:
            np.save(output + self._name + '_epoch_{}_att-weights.npy'.format(self.architecture, self.split_participants,
                                                                             self.p_rnn_hdim, epoch),
                    self.att_weights.cpu().numpy())

    def get_total_parameters(self):
        return sum(param.numel() for param in self.parameters())

    def get_attention_weights(self):
        return self.att_weights

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name
