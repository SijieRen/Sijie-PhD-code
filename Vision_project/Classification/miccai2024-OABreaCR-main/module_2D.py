import torch
import torch.nn as nn
import torch.nn.functional as nnf

class AttentionAlign(nn.Module):
    def __init__(self, in_planes, out_planes=3, stride=1, downsample=None):
        super().__init__()

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)

        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        print(self.grid.shape)
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

class AttentionPool(nn.Module):

    def __init__(self, in_channels=512):
        super(AttentionPool, self).__init__()

        self.attention_fc = nn.Conv2d(in_channels=in_channels, out_channels=1,
                                      kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        '''
        args: 
            - x: tensor of shape (B, C, H, W)
        returns:
            - attention_scores: tensor of shape (B, C, H, W)
        '''
        B, C, H, W = x.size()
        attention_scores = self.attention_fc(x).view(B, 1, H*W) #B, 1, HW
        attention_scores = self.softmax(attention_scores).view(B, 1, H, W)

        return attention_scores

if __name__ == "__main__":
    # example for SpatialTransformer
    model = SpatialTransformer(size=[32, 32])
    input1 = torch.randn(4, 32, 32, 32)
    input2 = torch.randn(4, 2, 32, 32)
    output = model(input1, input2)
    print(output.shape)
    
    # example for AttentionPool
    in_tensor = torch.randn(4, 32, 32, 32)
    model = AttentionPool(in_channels=32)
    output = model(in_tensor)
    print(output.shape)