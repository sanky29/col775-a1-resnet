import torch 
import math
import pdb
from torch.nn import Conv2d, Linear, Module, BatchNorm2d, AvgPool2d
from torch.nn import ModuleList, ReLU, Softmax, Parameter


CHANNELS = [16, 16, 32,64]
IMAGE_SIZES = [32,32,16,8]


class LayerNorm(Module):

    def __init__(self, dim, eps = 1e-5, momentum = 0.1):

        super(LayerNorm, self).__init__()
        
        #x = x_norm*alpha + beta
        self.alpha = Parameter(torch.ones(dim))
        self.beta = Parameter(torch.zeros(dim))
        
        #hyper parameter
        self.eps = eps
        self.dim = [-1*(i+1) for i in range(len(dim))]
        
    def forward(self, x):
        '''
        Args:
            x: B x C x H x W
        Reurns:
            x_hat: B x C x H x W
        '''
        #mean: B x 1 x 1 x 1
        #std: B x 1 x 1 x 1
        
        mean = x.mean(self.dim, keepdim = True)
        std = ((x - mean)**2).mean(self.dim, keepdim = True)

        #pertorm norm
        x = (x-mean)/(std + self.eps)**0.5
        x = x*self.alpha + self.beta

        return x


class BatchNorm(Module):

    def __init__(self, dim, eps = 1e-5, momentum = 0.1):

        super(BatchNorm, self).__init__()
        
        #x = x_norm*alpha + beta
        self.alpha = Parameter(torch.ones(dim).unsqueeze(-1).unsqueeze(-1))
        self.beta = Parameter(torch.zeros(dim).unsqueeze(-1).unsqueeze(-1))
        
        #the mean and std
        mean = torch.zeros(dim).unsqueeze(-1).unsqueeze(-1)
        std = torch.ones(dim).unsqueeze(-1).unsqueeze(-1)

        #register the tensors
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        
        #hyper parameter
        self.eps = eps
        self.momentum = momentum
    
    def forward(self, x):
        '''
        Args:
            x: B x C x H x W
        Reurns:
            x_hat: B x C x H x W
        '''
        #mean: C x 1 x 1
        #std: C x 1 x 1
        if(not self.train):
            mean = self.mean
            std = self.std
        else:
            mean = x.mean((0,2,3), keepdim = True)[0]
            std = ((x - mean)**2).mean((0,2,3), keepdim = True)[0]

        #pertorm norm
        x = (x-mean)/(std + self.eps)**0.5
        x = x*self.alpha + self.beta

        if(self.train):
            #accumulate stat
            self.mean = self.momentum*mean + (1-self.momentum)*self.mean
            self.std = self.momentum*std + (1-self.momentum)*self.std
            
        return x

class InstanceNorm(Module):

    def __init__(self, dim, eps = 1e-5):

        super(InstanceNorm, self).__init__()
        
        #x = x_norm*alpha + beta
        self.alpha = Parameter(torch.ones(dim).unsqueeze(-1).unsqueeze(-1))
        self.beta = Parameter(torch.zeros(dim).unsqueeze(-1).unsqueeze(-1))
        
        #hyper parameter
        self.eps = eps

    def forward(self, x):
        '''
        Args:
            x: B x C x H x W
        Reurns:
            x_hat: B x C x H x W
        '''
        #mean: B x C x 1 x 1
        #std: B x C x 1 x 1
        mean = x.mean((2,3), keepdim = True)
        std = ((x - mean)**2).mean((2,3), keepdim = True)

        #perform norm
        x = (x-mean)/(std + self.eps)**0.5
        x = x*self.alpha + self.beta
        
        return x


class GroupNorm(Module):

    def __init__(self, dim, ngrps = 8, eps = 1e-5):

        super(GroupNorm, self).__init__()
        
        #x = x_norm*alpha + beta
        self.alpha = Parameter(torch.ones(dim).unsqueeze(-1).unsqueeze(-1))
        self.beta = Parameter(torch.zeros(dim).unsqueeze(-1).unsqueeze(-1))
        
        #hyper parameter
        self.eps = eps
        self.g = ngrps

    def forward(self, x):
        '''
        Args:
            x: B x C x H x W
        Reurns:
            x_hat: B x C x H x W
        '''

        #x: B x G x [C/G] x H x W
        B,C,H,W = x.shape
        x = x.view(-1,self.g, C//self.g, H, W)

        #mean: B x G x 1 x 1 x 1
        #std: B x G x 1 x 1 x 1
        mean = x.mean((2,3,4), keepdim = True)
        std = ((x - mean)**2).mean((2,3,4), keepdim = True)

        #perform norm
        x = (x-mean)/(std + self.eps)**0.5
        
        x = x.view(B,C,H,W)
        x = x*self.alpha + self.beta
        
        return x


class BatchInstanceNorm(Module):

    def __init__(self, dim, eps = 1e-5, momentum = 0.1):

        super(BatchInstanceNorm, self).__init__()
        
        #x = x_norm*alpha + beta
        self.alpha = Parameter(torch.ones(dim).unsqueeze(-1).unsqueeze(-1))
        self.beta = Parameter(torch.zeros(dim).unsqueeze(-1).unsqueeze(-1))
        self.rho = Parameter(torch.zeros(dim).unsqueeze(-1).unsqueeze(-1))
        
        #the mean and std
        mean = torch.zeros(dim).unsqueeze(-1).unsqueeze(-1)
        std = torch.ones(dim).unsqueeze(-1).unsqueeze(-1)

        #register the tensors
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        
        #hyper parameter
        self.eps = eps
        self.momentum = momentum
    
    def forward(self, x):
        '''
        Args:
            x: B x C x H x W
        Reurns:
            x_hat: B x C x H x W
        '''

        #clamp the value
        self.rho.data = torch.clamp(self.rho.data, 0, 1.0)

        #mean_i: B x C x 1 x 1
        #std_i: B x C x 1 x 1
        mean_i = x.mean((2,3), keepdim = True)
        std_i = ((x - mean_i)**2).mean((2,3), keepdim = True)

        #mean_b: C x 1 x 1
        #std_b: C x 1 x 1

        if(not self.train):
            mean_b = self.mean
            std_b = self.std
        else:
            mean_b = x.mean((0,2,3), keepdim = True)[0]
            std_b = ((x - mean_b)**2).mean((0,2,3), keepdim = True)[0]

        #pertorm norm
        x_b = (x-mean_b)/(std_b + self.eps)**0.5
        x_i = (x-mean_i)/(std_i + self.eps)**0.5
        
        #final x
        x = self.rho*x_b + (1-self.rho)*x_i

        if(not self.train):
            #accumulate stat
            self.mean = self.momentum*mean_b + (1-self.momentum)*self.mean
            self.std = self.momentum*std_b + (1-self.momentum)*self.std

        return x

'''
this helper module creates stack of n conv layers with skip connection
'''
class ConvBlock(Module):

    def __init__(self, 
        n, 
        in_channels = 16, 
        out_channels = 16, 
        kernel_size = 3, 
        norm = BatchNorm2d, 
        input_shape = 32, 
        output_shape = 32,
        norm_type = None):

        super(ConvBlock, self).__init__()

        #the required zero padding
        zero_padding = (kernel_size - 1)//2

        #list of the conv layers
        '''
        Note that o = floor( (n - f + 2p)/s  + 1)
        '''
        stride = int(input_shape/output_shape)
        padding = math.ceil(((output_shape-1)*stride + kernel_size - input_shape)/2) 
        
        #shape mapping convolution
        start_layer = Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding)
        
        self.conv_layers = ModuleList([Conv2d(out_channels, out_channels, kernel_size = kernel_size, padding = zero_padding) for i in range(2*n-1)])
        self.conv_layers.insert(0, start_layer)
        
        #the required batch norm layers
        if(norm_type == 'ln'):
            self.norms = ModuleList([LayerNorm([out_channels, output_shape, output_shape]) for i in range(2*n)])    
        elif(norm is not None):
            self.norms = ModuleList([norm(out_channels) for i in range(2*n)])
        else:
            self.norms = [lambda x: x for i in range(2*n)]
            
        #activation
        self.relu = ReLU()
        self.n = n

    def forward(self, x, x_id):
        '''
        Args:
            x: B x IMAGE_SIZES[i-1] x IMAGE_SIZES[i-1] x in_channels
            x_id: B x IMAGE_SIZES[i] x IMAGE_SIZES[i] x out_channels (id mapping from previous block)
        Retruns:    
            output: B x IMAGE_SIZES[i] x IMAGE_SIZES[i] x out_channels
        '''
        output = x
        for i in range(0,self.n):
            
            #output: B x IMAGE_SIZES[i] x IMAGE_SIZES[i] x out_channels
            output = self.conv_layers[2*i](output)
            output = self.norms[2*i](output)
            output = self.relu(output)
            
            #output: B x IMAGE_SIZES[i] x IMAGE_SIZES[i] x out_channels
            output = self.conv_layers[2*i+1](output)
            output = self.norms[2*i+1](output)
            
            #H(x) = x + F(x)
            output = output + x_id
            output = self.relu(output)

            x_id = output
        return output
'''
the architecture of resnet is simple
image -> 
3x3 conv (16) -> 
3x3 conv (16) -> --- 2n such layers with skip connections
3x3 conv (32) -> --- 2n such layers with skip connections
3x3 conv (64) -> --- 2n such layers with skip connections
fc (10)
'''


class ResNet(Module):

    def __init__(self, n, r, norm = 'inbuilt'):
        super(ResNet, self).__init__()

        #the n layers of the 
        self.norm_type = norm

        #the norm class
        norm = self.get_normalization(norm)

        self.conv1 = Conv2d(in_channels = 3, out_channels = 16, kernel_size= 3, padding = 1)
        
        if(self.norm_type == 'ln'):
            self.norm1 = LayerNorm([16,32,32])
        elif(norm is None):
            self.norm1 = lambda x: x
        else:
            self.norm1 = norm(16)
        
        self.relu = ReLU()

        #the 32 x 32 x 16 block
        self.block_1 = ConvBlock( n, 
            in_channels = CHANNELS[0], 
            out_channels = CHANNELS[1], 
            kernel_size = 3, 
            norm = norm,
            input_shape = IMAGE_SIZES[0],
            output_shape = IMAGE_SIZES[1],
            norm_type = self.norm_type)

        #the skip connection from block_1 last second last layer
        #to first layer of block 2
        self.skip12 = Conv2d(in_channels = CHANNELS[1], out_channels = CHANNELS[2], kernel_size = 1, stride = 2)
        
        if(self.norm_type == 'ln'):
            self.norm12 = LayerNorm([CHANNELS[2],IMAGE_SIZES[2],IMAGE_SIZES[2]])
        elif(norm is None):
            self.norm12 = lambda x: x
        else:
            self.norm12 = norm(CHANNELS[2])


        #the 16 x 16 x 32 block
        self.block_2 = ConvBlock( n, 
            in_channels = CHANNELS[1], 
            out_channels = CHANNELS[2], 
            kernel_size = 3, 
            norm = norm,
            input_shape = IMAGE_SIZES[1],
            output_shape = IMAGE_SIZES[2],
            norm_type = self.norm_type)

        #the skip connection from block_2 last second last layer
        #to first layer of block 3
        self.skip23 = Conv2d(in_channels = CHANNELS[2], out_channels = CHANNELS[3], kernel_size = 1, stride = 2)      
        
        if(self.norm_type == 'ln'):
            self.norm23 = LayerNorm([CHANNELS[3],IMAGE_SIZES[3],IMAGE_SIZES[3]])
        elif(norm is None):
            self.norm23 = lambda x: x
        else:
            self.norm23 = norm(CHANNELS[3])
            

        #the 8 x 8 x 64 block
        self.block_3 = ConvBlock( n, 
            in_channels = CHANNELS[2], 
            out_channels = CHANNELS[3], 
            kernel_size = 3, 
            norm = norm,
            input_shape = IMAGE_SIZES[2],
            output_shape = IMAGE_SIZES[3],
            norm_type = self.norm_type)

        #the average pool
        average_pool_kernel_size = IMAGE_SIZES[-1]
        self.average_pool = AvgPool2d(average_pool_kernel_size)

        #the final fc layer
        self.fc = Linear(CHANNELS[3], r)

        #the softmax layer
        self.softmax = Softmax(dim = -1)

    def get_normalization(self, norm):
        if(norm == 'inbuilt'):
            return BatchNorm2d
        if(norm == 'none'):
            return None
        if(norm == 'bn'):
            return BatchNorm
        if(norm == 'in'):
            return InstanceNorm
        if(norm == 'bin'):
            return BatchInstanceNorm
        if(norm == 'ln'):
            return LayerNorm
        if(norm == 'gn'):
            return GroupNorm
        

    def forward(self, x):
        '''
        Args:
            x: B x 3 x H x W
        Output:
            y: B x R (probabilities of class)
        '''
        
        #x: B x CHANNELS[0] x H x W
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        #x: B x CHANNELS[1] x IMAGE_SIZES[1] x IMAGE_SIZES[1]
        x = self.block_1(x,x)

        #x: B x CHANNELS[2] x IMAGE_SIZES[2] x IMAGE_SIZES[2]
        x_id = self.skip12(x)
        x_id = self.norm12(x_id)
        
        #x: B x CHANNELS[2] x IMAGE_SIZES[2] x IMAGE_SIZES[2]
        x = self.block_2(x,x_id)

        #x: B x CHANNELS[3] x IMAGE_SIZES[3] x IMAGE_SIZES[3]
        x_id = self.skip23(x)
        x_id = self.norm23(x_id)

        #x: B x CHANNELS[3] x IMAGE_SIZES[3] x IMAGE_SIZES[3]
        x = self.block_3(x,x_id)

        #x: B x CHANNELS[3] x 1 x 1
        x = self.average_pool(x)
        
        #x: B x CHANNELS[3]
        x = x.squeeze(-1).squeeze(-1)
        f = x
        
        #x: B x r
        x = self.fc(x)

        # #the softmax layer
        x = self.softmax(x)

        return x,f