import torch
import dynamic_network_architectures.architectures.unet as m

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels : int, out_channels : int, stride: int = 1 ) -> None:
        super().__init__()
        self.Conv_0 = torch.nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=True)
        self.Norm_0 = torch.nn.InstanceNorm3d(num_features=out_channels, affine=True)
        self.Activation_0 = torch.nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.Conv_1 = torch.nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.Norm_1 = torch.nn.InstanceNorm3d(num_features=out_channels, affine=True)
        self.Activation_1 = torch.nn.LeakyReLU(negative_slope=0.01, inplace=True)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.Conv_0(input)
        output = self.Norm_0(output)
        output = self.Activation_0(output)
        output = self.Conv_1(output)
        output = self.Norm_1(output)
        output = self.Activation_1(output)
        return output

class UNetHead(torch.nn.Module):
    def __init__(self, in_channels: int, nb_class: int) -> None:
        super().__init__()
        self.Conv = torch.nn.Conv3d(in_channels = in_channels, out_channels = nb_class, kernel_size = 1, stride = 1, padding = 0)
        self.Softmax = torch.nn.Softmax(dim=1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.Conv(input)
        output = self.Softmax(output)
        output = torch.argmax(output, dim=1).unsqueeze(1)
        return output
    
class UNetBlock(torch.nn.Module):

    def __init__(self, channels, nb_class: int, mr : bool , lung: bool, i : int = 0) -> None:
        super().__init__()

        if i>0:
            if not lung:
                self.DownConvBlock = ConvBlock(in_channels=channels[0], out_channels=channels[1], stride= (1,2,2) if mr and i > 4 else 2)
            else:
                self.DownConvBlock = ConvBlock(in_channels=channels[0], out_channels=channels[1], stride= (2,1,2) if i > 4 else 2)
        else:
            self.DownConvBlock = ConvBlock(in_channels=channels[0], out_channels=channels[1], stride=1)
     
        if len(channels) > 2:
            self.UNetBlock = UNetBlock(channels[1:], nb_class, mr, lung, i+1)
            self.UpConvBlock = ConvBlock(in_channels=channels[1]*2, out_channels=channels[1])
            self.Head = UNetHead(channels[1], nb_class)
            
        if i > 0:
            if not lung:
                self.CONV_TRANSPOSE = torch.nn.ConvTranspose3d(in_channels = channels[1], out_channels = channels[0], kernel_size =  (1,2,2) if mr and i > 4 else 2, stride = (1,2,2) if mr and i > 4 else 2 , padding = 0)
            else:
                self.CONV_TRANSPOSE = torch.nn.ConvTranspose3d(in_channels = channels[1], out_channels = channels[0], kernel_size =  (2,1,2) if i > 4 else 2, stride = (2,1,2) if i > 4 else 2 , padding = 0)
  
    def forward(self, input: torch.Tensor, i: int = 0) :
        layers = []
        output = input
        output = self.DownConvBlock(output)

        if hasattr(self, "UNetBlock"):
            output, ls = self.UNetBlock(output, i+1)
            for l in ls:
                layers.append(l)
            output = self.UpConvBlock(output)
            layers.append(self.Head(output))

        if i > 0:
            output = self.CONV_TRANSPOSE(output)
            output = torch.cat((output, input), dim=1)
        return output, layers

class Unet_TS(torch.nn.Module):

    def __init__(self, channels = [1, 64, 128, 256, 512, 1024], nb_class: int = 2,  mr: bool = False, lung: bool = False) -> None:
        super().__init__()
        self.UNetBlock = UNetBlock(channels, nb_class, mr, lung)

    def forward(self, input: torch.Tensor):
        _, layers = self.UNetBlock(input)
        return layers[-1]

class ResidualEncoderUNet(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.model = m.ResidualEncoderUNet(input_channels = 1,
                          n_stages = 6,
                          features_per_stage = [32, 64, 128, 256, 320, 320],
                          conv_op = torch.nn.modules.conv.Conv3d,
                          kernel_sizes = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
                          strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]],
                          n_blocks_per_stage = [1, 3, 4, 6, 6, 6],
                          num_classes = 3,
                          n_conv_per_stage_decoder = [1, 1, 1, 1, 1],
                          conv_bias = True,
                          norm_op = torch.nn.modules.instancenorm.InstanceNorm3d,
                          norm_op_kwargs =  {'eps': 1e-05, 'affine': True},
                          dropout_op = None,
                          dropout_op_kwargs = None,
                          nonlin = torch.nn.LeakyReLU, 
                          nonlin_kwargs= {'inplace': True})
        
    def forward(self, input: torch.Tensor):
        return torch.argmax(self.model(input.to(torch.float32)), dim=1).unsqueeze(1).float()