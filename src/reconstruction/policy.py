import os

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
            
        # Network settings
        self.rnn_hidden_size = 256
        self.iscuda = True
        # The input size for location embedding / proprioception stack
        self.input_size_loc = 256 # Relative camera position
        self.input_size_act = 256

        # (1) Sense image: Takes in BxCx32x32 image input and converts it to Bx256 matrix.
        self.sense_im = nn.Sequential(self._conv(3, 16),      # Bx16x16x16
                                      self._conv(16, 32),     # Bx32x8x8
                                      self._conv(32, 64),     # Bx64x4x4
                                      View((-1, 1024)),       # Bx1024
                                      self._linear(1024, 256) # Bx256
                                     )
        
        # NOTE: EXPERIMENTAL. Try implementing a view history.
        self.sense_views = self._linear(self.input_size_act, self.input_size_act)

        # (2) Sense proprioception stack: Converts proprioception inputs to 16-D vector.
        self.sense_pro = self._linear(self.input_size_loc, 16)

        # (3) Fuse: Fusing the outputs of (1) and (2) to give 256-D vector per image
        # May be appropriate to add activation function later or change to self._linear.
        self.fuse = nn.Sequential(self._linear(528, 256), # Bx256 # NOTE: THIS USED TO BE 272.
                                  nn.Linear(256, 256),    # Bx256
                                  nn.BatchNorm1d(256)
                                 )

        # (4) Aggregator: View aggregating LSTM
        self.aggregate = nn.LSTM(input_size=256, hidden_size=self.rnn_hidden_size, num_layers=1)

        # (5) Act module: Takes in aggregator hidden state to produce probability distribution over actions 
        self.act = nn.Sequential(self._linear(self.input_size_act, 128),
                                 self._linear(128, 128),
                                 nn.Linear(128, 256)    # because (512/32)**2=256
                                )
        
        # (6) Decode module: Given the current representation of the image, reconstruct the full view.
        self.decode = nn.Sequential(self._linear(256, 1024), # Bx1024
                                    View((-1, 64, 4, 4)),    # Bx64x4x4
                                    self._deconv(64, 64),    # Bx64x8x8
                                    self._deconv(64, 32),    # Bx32x16x16
                                    self._deconv(32, 32),    # Bx32x32x32
                                    self._deconv(32, 16),    # Bx16x64x64
                                    self._deconv(16, 16),    # Bx16x128x128
                                    self._deconv(16, 8),     # Bx8x256x256
                                    nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1), # Bx3x512x512
                                    nn.Sigmoid()
                                   )
 
            
    def forward(self, x, views, hidden=None):
        
        # "Sense" the image.
        x1 = self.sense_im(x['im']) #.squeeze()
        x2 = self.sense_pro(x['pro'])
        x3 = self.sense_views(views) # NOTE: EXPERIMENTAL.
        x = torch.cat([x1, x2, x3], dim=1)
        batch_size = x.shape[0]
        
        # Set up the recurrent hidden layers.
        # NOTE: confused about why there are two hidden layers here. Maybe that's just how RNN's work in PyTorch?
        if hidden is None:
            hidden = [Variable(torch.zeros(1, batch_size, self.rnn_hidden_size)), # hidden state: (num_layers, batch_size, hidden size)
                      Variable(torch.zeros(1, batch_size, self.rnn_hidden_size))] # cell state  :(num_layers, batch_size, hidden size)
            if self.iscuda:
                hidden[0] = hidden[0].cuda()
                hidden[1] = hidden[1].cuda()
            
        # Fuse the proprioceptive representation and the view representation.
        x = self.fuse(x)

        # Update the belief state about the image.
        # Note: input to aggregate lstm has to be (seq_length x batch_size x input_dims)
        # Since we are feeding in the inputs one by one, it is 1 x batch_size x 256
        x, hidden = self.aggregate(x.view(1, *x.size()), hidden)
        
        # Define input to the action and decoding layers.
        act_input = hidden[0].view(batch_size, -1)

        # Predict the probability of all actions.
        probs = F.softmax(self.act(act_input), dim=1)
            
        # Decode the whole image using the decoder.
        decoded = self.decode(act_input)
        return probs, hidden, decoded
    
    def _linear(self, in_size, out_size):
        return nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.BatchNorm1d(out_size),
            nn.ReLU(inplace=True)
        )
            
    def _conv(self, in_size, out_size):
        return nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def _deconv(self, in_size, out_size):
        return nn.Sequential(
            nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

'''
Module to reshape a torch tensor in nn.Sequential.
'''
class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)