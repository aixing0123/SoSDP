import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
TaskNum=5
distcon=2
const1=0.96


class Encoder(nn.Module):
    """Encodes the static & dynamic states using 1d Convolution."""

    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size=1)

    def forward(self, input):
        output = self.conv(input)
        return output  # (batch, hidden_size, seq_len)


class Attention(nn.Module):
    """Calculates attention over the input nodes given the current state."""

    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        # W processes features from static decoder elements
        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size),
                                          device=device, requires_grad=True))

        self.W = nn.Parameter(torch.zeros((1, hidden_size, 3 * hidden_size),
                                          device=device, requires_grad=True))

    def forward(self, static_hidden, dynamic_hidden, decoder_hidden):

        batch_size, hidden_size, _ = static_hidden.size()

        hidden = decoder_hidden.unsqueeze(2).expand_as(static_hidden)
        hidden = torch.cat((static_hidden, dynamic_hidden, hidden), 1)

        # Broadcast some dimensions so we can do batch-matrix-multiply
        v = self.v.expand(batch_size, 1, hidden_size)
        W = self.W.expand(batch_size, hidden_size, -1)

        attns = torch.bmm(v, torch.tanh(torch.bmm(W, hidden)))
        attns = F.softmax(attns, dim=2)  # (batch, seq_len)
        return attns


class Pointer(nn.Module):
    """Calculates the next state given the previous state and input embeddings."""

    def __init__(self, hidden_size, num_layers=1, dropout=0.2):
        super(Pointer, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Used to calculate probability of selecting next state
        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size),
                                          device=device, requires_grad=True))

        self.W = nn.Parameter(torch.zeros((1, hidden_size, 2 * hidden_size),
                                          device=device, requires_grad=True))

        # Used to compute a representation of the current decoder output
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)
        self.encoder_attn = Attention(hidden_size)

        self.drop_rnn = nn.Dropout(p=dropout)
        self.drop_hh = nn.Dropout(p=dropout)

    def forward(self, static_hidden, dynamic_hidden, decoder_hidden, last_hh):

        rnn_out, last_hh = self.gru(decoder_hidden.transpose(2, 1), last_hh)
        rnn_out = rnn_out.squeeze(1)

        # Always apply dropout on the RNN output
        rnn_out = self.drop_rnn(rnn_out)
        if self.num_layers == 1:
            # If > 1 layer dropout is already applied
            last_hh = self.drop_hh(last_hh) 

        # Given a summary of the output, find an  input context
        enc_attn = self.encoder_attn(static_hidden, dynamic_hidden, rnn_out)
        context = enc_attn.bmm(static_hidden.permute(0, 2, 1))  # (B, 1, num_feats)

        # Calculate the next output using Batch-matrix-multiply ops
        context = context.transpose(1, 2).expand_as(static_hidden)
        energy = torch.cat((static_hidden, context), dim=1)  # (B, num_feats, seq_len)

        v = self.v.expand(static_hidden.size(0), -1, -1)
        W = self.W.expand(static_hidden.size(0), -1, -1)

        probs = torch.bmm(v, torch.tanh(torch.bmm(W, energy))).squeeze(1)

        return probs, last_hh


class DRL4SoS(nn.Module):
    """Defines the main Encoder, Decoder, and Pointer combinatorial models.

    Parameters
    ----------
    static_size: int
        Defines how many features are in the static elements of the model
        (e.g. 2 for (x, y) coordinates)
    dynamic_size: int > 1
        Defines how many features are in the dynamic elements of the model
        (e.g. 2 for the VRP which has (load, demand) attributes. The SoS doesn't
        have dynamic elements, but to ensure compatility with other optimization
        problems, assume we just pass in a vector of zeros.
    hidden_size: int
        Defines the number of units in the hidden layer for all static, dynamic,
        and decoder output units.
    update_fn: function or None
        If provided, this method is used to calculate how the input dynamic
        elements are updated, and is called after each 'point' to the input element.
    mask_fn: function or None
        Allows us to specify which elements of the input sequence are allowed to
        be selected. This is useful for speeding up training of the networks,
        by providing a sort of 'rules' guidlines to the algorithm. If no mask
        is provided, we terminate the search after a fixed number of iterations
        to avoid tours that stretch forever
    num_layers: int
        Specifies the number of hidden layers to use in the decoder RNN
    dropout: float
        Defines the dropout rate for the decoder
    """

    def __init__(self, static_size, dynamic_size, hidden_size,
                 update_fn=None, mask_fn=None, num_layers=1, dropout=0.):
        super(DRL4SoS, self).__init__()

        if dynamic_size < 1:
            raise ValueError(':param dynamic_size: must be > 0, even if the '
                             'problem has no dynamic elements')

        self.update_fn = update_fn
        self.mask_fn = mask_fn

        # Define the encoder & decoder models
        self.static_encoder = Encoder(static_size-4, hidden_size)
        self.dynamic_encoder = Encoder(dynamic_size, hidden_size)
        self.decoder = Encoder(static_size-4, hidden_size)
        self.pointer = Pointer(hidden_size, num_layers, dropout)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

        # Used as a proxy initial state in the decoder when not specified
        self.x0 = torch.zeros((1, static_size-4, 1), requires_grad=True, device=device)

    def forward(self, static, dynamic, decoder_input=None, last_hh=None):
        """
        Parameters
        ----------
        static: Array of size (batch_size, feats, num_cities)
            Defines the elements to consider as static. For the SoS, this could be
            things like the (x, y) coordinates, which won't change
        dynamic: Array of size (batch_size, feats, num_cities)
            Defines the elements to consider as static. For the VRP, this can be
            things like the (load, demand) of each city. If there are no dynamic
            elements, this can be set to None
        decoder_input: Array of size (batch_size, num_feats)
            Defines the outputs for the decoder. Currently, we just use the
            static elements (e.g. (x, y) coordinates), but this can technically
            be other things as well
        last_hh: Array of size (batch_size, num_hidden)
            Defines the last hidden state for the RNN
        """

        batch_size, input_size, sequence_size = static.size()

        if decoder_input is None:
            decoder_input = self.x0.expand(batch_size, -1, -1)

        # Always use a mask - if no function is provided, we don't update it
        mask = torch.ones(batch_size, sequence_size, device=device)

        for i in range(batch_size):
            #chose_idx1=torch.ones(mask.shape[0],TaskNum, device=device).long()
            for j in range(mask.shape[1]):
                coord1=static[i,11:,j].cpu().numpy()
                coord2=static[i,9:11,j].cpu().numpy()
                dist01=np.sqrt(np.sum(np.square(coord1-coord2)))
                # print(dist01)
                if dist01>distcon:
                    mask[i,j]=0

                
   
        # Structures for holding the output sequences
        tour_idx, tour_logp = [], []
        tour_idx=torch.tensor(tour_idx,device=device).int()
        tour_logp=torch.tensor(tour_logp,device=device).int()
        max_steps = sequence_size if self.mask_fn is None else 1000

        # Static elements only need to be processed once, and can be used across
        # all 'pointing' iterations. When / if the dynamic elements change,
        # their representations will need to get calculated again.
        static_hidden = self.static_encoder(static[:,:9,:])
        dynamic_hidden = self.dynamic_encoder(dynamic)
 
        for i in range(batch_size):
            tour_idx1, tour_logp1 = [], []

            t=static[i,:4,:5].cpu().numpy().T
            sysnum=np.zeros(TaskNum)

            
            # print(t)
            # t.append(T1.numpy())
            # t.append(T2.numpy())
            # t.append(T3.numpy())
            # t.append(T4.numpy())
            # t.append(T5.numpy())

            for _ in range(max_steps):
                

               
                # print(np.array(t))
                if np.array(t).sum()== 0 or np.array(mask[i].cpu().numpy()).sum()== 0:
                    # print("hhhh")
                    for j in range(sequence_size//TaskNum - _):
                        tour_logp1.append(torch.tensor([0],device=device).unsqueeze(1))
                        tour_idx1.append(torch.tensor([-1],device=device).unsqueeze(1))
                    break
                if not mask[i].byte().any():
                    break
                # ... but compute a hidden rep for each element added to sequence
                decoder_hidden = self.decoder(decoder_input)

                probs, last_hh = self.pointer(static_hidden[i].unsqueeze(0),
                                            dynamic_hidden[i].unsqueeze(0),
                                            decoder_hidden[i].unsqueeze(0), last_hh)
                probs = F.softmax(probs + mask[i].unsqueeze(0).log(), dim=1)

                # When training, sample the next step according to its probability.
                # During testing, we can take the greedy approach and choose highest
                if self.training:
                    m = torch.distributions.Categorical(probs)

                    # Sometimes an issue with Categorical & sampling on GPU; See:
                    # https://github.com/pemami4911/neural-combinatorial-rl-pytorch/issues/5
                    ptr = m.sample()
                    while not torch.gather(mask[i].unsqueeze(0), 1, ptr.data.unsqueeze(1)).byte().all():
                        ptr = m.sample()
                    logp = m.log_prob(ptr)
                else:
                    prob, ptr = torch.max(probs, 1)  # Greedy
                    logp = prob.log()

                # After visiting a node update the dynamic representation
                if self.update_fn is not None:
                    dynamic = self.update_fn(dynamic, ptr.data)
                    dynamic_hidden = self.dynamic_encoder(dynamic)

                    # Since we compute the VRP in minibatches, some tours may have
                    # number of stops. We force the vehicles to remain at the depot 
                    # in these cases, and logp := 0
                    is_done = dynamic[:, 1].sum(1).eq(0).float()
                    logp = logp * (1. - is_done)

                n2=ptr.data.cpu().numpy()%TaskNum 
                n2=n2[0]
                n3=ptr.data.cpu().numpy()[0]
                n1=-1
                sysnum[n2]+=1

                # print(ptr.data.cpu().numpy()[0])
                for j in range(4):
                    # print(t[n2][j])
                    # print(static[i][3+j].cpu().numpy())
                    t[n2][j]=t[n2][j]-static[i][4+j][n3].cpu().numpy()*(const1**(sysnum[n2]))
                    #print("sysnum[j]、static[i][4+j][n3]、static[i][4+j][n3].cpu().numpy()**(sysnum[j]为：",sysnum[n2],static[i][4+j][n3].cpu().numpy(),static[i][4+j][n3].cpu().numpy()*(const1**(sysnum[n2])))
                    if t[n2][j]<0:
                        t[n2][j]=0
                if np.array(t[n2]).sum()==0:
                    n1=n2
                # And update the mask so we don't re-visit if we don't need to
                if self.mask_fn is not None:
                    mask[i]= self.mask_fn(mask[i].unsqueeze(0), dynamic[i].unsqueeze(0), ptr.data.unsqueeze(0),n1).squeeze(0).detach()

                tour_logp1.append(logp.unsqueeze(1))
                tour_idx1.append(ptr.data.unsqueeze(1))

                decoder_input= torch.gather(static[:,:9,:], 2,ptr.expand(batch_size,1).view(-1, 1, 1).expand(-1, input_size-4, 1)).detach()
            # print(tour_idx1.size)
            tour_idx1 = torch.cat(tour_idx1, dim=1)  
            tour_logp1 = torch.cat(tour_logp1, dim=1)

            
            tour_idx = torch.cat((tour_idx,tour_idx1), dim=0)
            tour_logp = torch.cat((tour_logp,tour_logp1), dim=0)


        return tour_idx, tour_logp


if __name__ == '__main__':
    raise Exception('Cannot be called from main')
