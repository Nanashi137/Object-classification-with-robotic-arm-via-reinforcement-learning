import torch 
import torch.nn as nn 


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int=1, downsample=None) -> None:
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3), stride=stride, padding=(1,1), bias=False)
        self.bn1   = nn.BatchNorm2d(num_features=out_channels)
        self.relu  = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3,3), stride=1, padding=(1,1), bias=False)
        self.bn2   = nn.BatchNorm2d(num_features=out_channels)

        self.downsample = downsample

    def forward(self, x: torch.Tensor): 
        identity = x

        out = self.conv1(x) 
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample != None: 
            identity = self.downsample(x)
        
        out += identity 
        out = self.relu(out)

        return out



class Resnet10(nn.Module): 
    def __init__(self, height: int, width: int, mode: str="gray") -> None:
        super(Resnet10, self).__init__()

        self.height = height 
        self.width = width 
        self.image_channels = 1 if mode == "gray" else 3
        self.in_channels=64
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=self.image_channels, out_channels=64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=(1,1)),
        )

        self.layer1 = self._make_layers(out_channels=64, n_blocks=2, stride=1)
        self.layer2 = self._make_layers(out_channels=128, n_blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.hidden_dim_cnn = list(self.layer2.children())[-1].bn2.num_features

        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim_cnn, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.hidden_dim_mlp = list(self.mlp.children())[-2].num_features

    def _make_layers(self, out_channels: int, n_blocks: int, stride: int): 
        downsample = None
        if stride != 1: 
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels, kernel_size=(1,1), stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = []
        layers.append(ResidualBlock(in_channels=self.in_channels, out_channels=out_channels, stride=stride, downsample=downsample))

        self.in_channels = out_channels
        for _ in range (1,n_blocks):
            layers.append(ResidualBlock(in_channels=out_channels, out_channels=out_channels))
        
        return nn.Sequential(*layers)



    def forward(self, x: torch.Tensor):
        out = self.conv(x)

        out = self.layer1(out)
        out = self.layer2(out)
        
        out = self.avgpool(out)

        out = torch.flatten(out, start_dim=1)
        out = self.mlp(out)

        return out
    
    def _count_params(self): 
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
if __name__=="__main__": 

    resblock = Resnet10(height=224, width=224, mode="gray")
    # print(resblock)
    # n_params = sum(p.numel() for p in resblock.parameters() if p.requires_grad)
    # print(f"Total parameters of Resnet10: {n_params}")
    resblock.eval()
    dummy = torch.randn(1,1,64,64)

    output= resblock(dummy)
    print(output.shape)
