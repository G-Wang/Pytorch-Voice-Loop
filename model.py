import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
# padding functionalities for batching
from torch.nn.utils.rnn import pack_padded_sequence as unpack
from torch.nn.utils.rnn import pad_packed_sequence as pack
from utils import getlinear

class Encoder(nn.Module):
    """Encoder class.

    The enconder encodes the sentence phonemes (input) as well as speaker, and returns the phoneme(input)
    embedding as well as the speaker embedding

    Args:
        hparams(class): hyper-parameters
        device(device): device to put module on

    """
    def __init__(self, hparams, device):
        super().__init__()
        self.hp = hparams
        self.device = device
        self.to(device)
        # initialize embedding layers. See hparams class for more details on parameter naming
        self.Lut_P = nn.Embedding(self.hp.si, self.hp.dp) # phoneme embedding
        self.Lut_S = nn.Embedding(self.hp.ns, self.hp.ds) # speaker embedding

    def forward(self, sentence, speaker_id):
        """Given a sentence (or tuple of padded sentence and sentence length) and speaker_id, compute the embedding
        
        Args:
            sentence(batch_size x seq_len or tuple): Two types of sentences can be passed to the encoder, in the
                                                    first type, it's just a torch LongTensor of shape batch_size x seq_len. 
                                                    In the second type, we pass a tuple containing an already padded 
                                                    sentence (batch_size x max_seq_len) along with a torch tensor list of length
            
            speaker_id(batch_size x 1): speaker id list for batch of input data

        """
        # check to see if sentence is tuple
        if isinstance(sentence, tuple):
            assert sentence[0].shape[0] == len(sentence[1]), "padded data batch size does not match number of sequences"
            # embed padded sentence
            lut_p = self.Lut_P(sentence[0].to(self.device))
            # unpack lut_p with actual sequence length into packedsequence
            lut_p = unpack(lut_p, sentence[1].to(self.device), batch_first=True) # always batch first
            # pad our packed sequence, only take the actual padding, we don't need to sequence length
            lut_p = pack(lut_p, batch_first=True)[0]
        else:
            # else we just have a batch tensor of sentence
            lut_p = self.Lut_P(sentence.to(self.device))
        
        lut_s = self.Lut_S(speaker_id.to(self.device)).view(-1, self.hp.ds) # compute lut_s and reshape to batch_size x ds
        
        return lut_p, lut_s


class Attention(nn.Module):
    COEF = 0.3989422917366028  # numpy.sqrt(1/(2*numpy.pi))
    """GMM attetion. This module contains all the weights for GMM attentiona as well as computation for it.

    Attributes:
        Na (d*k x 3c): attention weight, maps from buffer to 3 components of GMM

    Args:
        hparams(class):
        device(device):
        
    """
    def __init__(self, hparams, device):
        super().__init__()
        self.hp = hparams
        self.device = device
        self.to(device)
        self.Na = getlinear(self.hp.d*self.hp.k, 3*self.hp.c)

    def forward(self, lut_p, S_tm1, mu_tm1):
        """forward pass through attention given lut_p, S_t-1, mu_t-1. compute the GMM outputs, which are the
        c_t, mu_t and alpha_t

        Args:
            lut_p (batch_size x seq_len x embedding_dim):
            S_tm1 (batch_size x d*k): buffer at time - 1
            mu_tm1 (batch_size x c): GMM mean from previous time step

        """
        batch_size = lut_p.shape[0]
        seq_len = lut_p.shape[1]
        # forward pass on flattened S_tm1 via Na and reshape
        out = self.Na(S_tm1.view(-1,self.hp.k*self.hp.d)).view(-1, 3, self.hp.c)
        # get attention model inputs
        k_t = out[:,0,:]
        b_t = out[:,1,:]
        y_t = out[:,2,:]
        # compute GMM parameters
        # assert k_t.shape == mu_tm1.shape, 'k_t and mu_tm1 shapes are not equal'
        mu_t = mu_tm1 + self.hp.attention_alignment*torch.exp(k_t)
        sig_t = torch.exp(b_t)
        g_t = F.softmax(y_t, dim=1)
        # compute GMM
        # initialize J
        J = torch.arange(1,seq_len+1).expand_as(torch.Tensor(batch_size,self.hp.c, seq_len)).to(self.device)
        # reshape GMM parameters
        g_t = g_t.unsqueeze(-1).expand(g_t.shape[0],g_t.shape[1],seq_len)
        sig_t = sig_t.unsqueeze(-1).expand_as(g_t)
        mu_t_ = mu_t.unsqueeze(-1).expand_as(g_t)
         # compute phi_t
        phi_t = g_t * torch.exp(-0.5*sig_t*(mu_t_ - J)**2)
        alpha_t = self.COEF*torch.sum(phi_t,1).unsqueeze(1)
        # c_t
        c_t = torch.bmm(alpha_t, lut_p).transpose(0,1).squeeze(0)
        
        return c_t, mu_t, alpha_t

class Decoder(nn.Module):
    """Decoder module. The decoder takes the output from the encoder and GMM attention module and computes and update
    the buffer.

    Attributes:

    Args:

    """
    def __init__(self, hparams, device):
        super().__init__()
        self.hp = hparams
        self.device = device
        self.to(device)
        self.N_u = getlinear(self.hp.d*self.hp.k, self.hp.d)
        self.N_o = getlinear(self.hp.d*self.hp.k, self.hp.do)
        self.F_u = nn.Linear(self.hp.ds, self.hp.dp)
        self.F_o = nn.Linear(self.hp.ds, self.hp.d*self.hp.k)

    def update_buffer(self, S_tm1, c_t, o_tm1, ident):
        """Update buffer
        Args:
            S_tm1(batch_size x d x k):
            c_t(batch_size x c):
            o_tm1 (batch_size x 1 x do):
            ident(batch_size x ds):

        Returns:
            S
            
        """
        idt = F.tanh(self.F_u(ident))
        o_tm1 = o_tm1.view(-1,self.hp.do)
        C_t = torch.cat([c_t+idt, o_tm1/30], dim=1) # was in their original code
        C_t = C_t.unsqueeze(2)
        Sp = torch.cat([C_t, S_tm1[:,:,:-1]],2)
        u = self.N_u(Sp.view(Sp.shape[0],-1))
        u = u.unsqueeze(2)
        S = torch.cat([u, S_tm1[:,:,:-1]],2)
        
        return S

    
    def forward(self, S_tm1, c_t, o_tm1, ident):
        """forward pass through decoder
        
        Args:
            S_tm1 (batch_size x d x k):
            c_t (batch_size x c):
            o_tm1 (batch_size x 1 x do):
            ident (batch_size x ds):

        Returns:
            S_t:
            o_t:

        """
        # udpate buffer
        S_t = self.update_buffer(S_tm1, c_t, o_tm1, ident)
        # compute add S_t and speaker projection
        summation = S_t.view(S_t.shape[0],-1) + self.F_o(ident)
        ot_out = self.N_o(summation)

        return S_t, ot_out.view(-1,1,self.hp.do)


class MaskMSE(nn.Module):
    """Masked MSE loss.

    """
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.to(device)
    
    def _mask(self, sequence_len, max_len=None):
        if max_len is None:
            max_len = sequence_len.max()
        
        batch_size = len(sequence_len)
        seq_range = torch.range(0, max_len-1).long().to(self.device)
        seq_range_expand = seq_range.expand_as(torch.Tensor(batch_size, max_len))    
        seq_length_expand = sequence_len.unsqueeze(1).expand_as(seq_range_expand)
        return (seq_range_expand < seq_length_expand).float()
    
    def forward(self, inputs, targets, seq_len):
        """Given an inputs, targets and seq_len representing the seq_len of the mask

        """
        mask = self._mask(seq_len)
        mask = mask.unsqueeze(2)
        # put mask and target to device
        mask = mask.expand_as(targets).to(self.device)
        targets = targets.to(self.device)
        mask_sum = float(mask.sum())
        loss = F.mse_loss(inputs*mask, targets, size_average=False)
        # average over mask sum
        loss = loss / (mask_sum+1e-8)
        return loss


class Loop(nn.Module):
    """Main Voice loop model.

    The loop model is composed of an encoder, attention and decoder modules.

    Attributes:

    """
    def __init__(self, hparams, device):
        super().__init__()
        self.hp = hparams
        self.device = device
        self.encoder = Encoder(hparams, device)
        self.attention = Attention(hparams, device)
        self.decoder = Decoder(hparams, device)
        self.loss_func = MaskMSE(device)
        self.to(device)

    def initialize(self, sentence, speaker_id):
        """Initialization function.

        This function should be called before passing a sentence/batch of sentences to the model. This function
        will proceed to initialize p_embedding, s_embedding, S_t and mu_t.

        Note that special speaker specific initialization will be done according to outlined in paper.

        Args:

            sentence(tuple or LongTensor of batch_size x max_seq_len):
            speaker_id (batch_size x 1):

        Returns:
            p_embedding(batch_size x max_seq_len x dp):
            s_embedding(batch_size x ds)
            S_t (batch_size x d x k):
            mu_t (batch_size x c):

        """
        p_embedding, s_embedding = self.encoder(sentence, speaker_id)
        batch_size = speaker_id.shape[0]
        # create S buffer
        S_t = torch.zeros(batch_size, self.hp.d, self.hp.k)
        # initialize buffer with speaker embedding as described Step III of paper
        S_t[:,:self.hp.dp,:] = s_embedding.unsqueeze(2).expand(s_embedding.shape[0], s_embedding.shape[1], self.hp.k)
        # also initialize our mu_tm1, the initial mean parmeter for our GMM
        mu_t = torch.zeros(batch_size, self.hp.c)
        return p_embedding, s_embedding, S_t.to(self.device), mu_t.to(self.device)

    def count_parameters(model):
        """Return total number of parameters in millions

        """
        num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return num/1e6


    def forward(self, p_embedding, s_embedding, o_tm1, S_tm1, mu_tm1):
        """Does one step of forward pass, this is equivalent to one step in a sequence.

        Args:
            p_embedding(batch_size x max_seq_len x dp): phoneme/input embedding
            s_embedding(batch_size x ds): speaker embedding
            o_tm1(batch_size x do): previous output
            S_tm1(batch_size x d x k): previous buffer
            mu_tm1(batch_size x c): previous attention mean

        Returns:
            o_t(batch_size x do): next output
            S_t(batch_size x d x k): next buffer
            mu_t(batch_size x c): next attention mean

        """
        # compute attention
        c_t, mu_t, alpha_t = self.attention(p_embedding, S_tm1, mu_tm1)
        # decoder forward pass, returns new buffer (S_t), output (o_t) and speaker projection (sp_t)
        S_t, o_t = self.decoder.forward(S_tm1, c_t, o_tm1, s_embedding)

        return o_t, S_t, mu_t

    def compute_loss_batch(self, sentence, speaker_id, target, teacher_forcing=False):
        """Compute loss for a batch of input/output data.

        Note this function has multiple ways of conditioning on the next sequence.

        For training, it's recommended to keep teacher_forcing to True, this will force teacher forcing to 
        follow 3.1 as outlined in the paper. One can set the teacher forcing noise in hparams

        For validation, it's recommended to keep teacher_forcing to False

        Args:
            sentence(tuple of batch_size x seq_len and batch_seq_len_list):
            speaker_id(batch_size x 1):
            target(batch_size x seq_len x output_dim):
            teacher_forcing(bool): if True, averaged teacher forcing will be implemented following paper

        """
        assert isinstance(sentence, tuple), "sentence provided is not tuple"
        assert isinstance(target, tuple), "target provided is not tuple"
        assert sentence[0].shape[0] == len(sentence[1]), 'batch size of sentence[0] does not match batch_size_len'
        assert target[0].shape[0] == len(target[1]), 'batch size of target[0] does not match batch_size_len'
        # model initialize and initialize o_t to zero
        p_embedding, s_embedding, S_t, mu_t = self.initialize(sentence, speaker_id)
        batch_size = speaker_id.shape[0]
        o_t = torch.zeros(batch_size, self.hp.do).to(self.device)
        # create output collection and get target data and length
        output_collect = []
        target_data = target[0].to(self.device)
        target_len = target[1].to(self.device)
        # get seq_len
        seq_len = target_data.shape[1]
        # loop through sequence
        for i in range(seq_len):
            # forward pass
            o_t, S_t, mu_t = self.forward(p_embedding, s_embedding, o_t, S_t, mu_t)
            # append output to output list
            output_collect.append(o_t)
            # prepare o_t for next step, teacher forcing if need to
            if teacher_forcing:
                # see section 3.1 of paper
                #o_t = (o_t + target_data[:,i,:].unsqueeze(1))/2+np.random.normal(0, self.hp.noise_range)
                # directly predict on the target sequence
                o_t = target_data[:,i,:].unsqueeze(1)+np.random.normal(0, self.hp.noise_range)
            else:
                # use output directly
                o_t = o_t

        # compute loss
        total_output = torch.cat(output_collect, dim=1)
        loss = self.loss_func(total_output, target_data, target_len)

        return loss