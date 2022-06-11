#ResNet可以分成多种亚结构 Res18 Res34 Res50 Res101 Res152

classs BasicBlock (nn.Module):
  expansion = 1
  #expansion这个对应在残差结构中我们主分支的卷积核的个数是否发生变化
  
  def __init__(self, in_channel, out_channel, stride = 1, downsample = None):
    #downsample这个结构对应的是虚线所对应的残差结构，输入和输出的尺寸不一致时需要对输入进行处理
    super(BasicBlock, self).__init__()
    self.con1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,kernel_size=3,stride=stride,padding=1,bias=False)
    self.bn1 = nn.BatchNorm2d(out_channel)
    self.relu = nn.ReLU
    self.conv2 = nn.Conv2d(in_channels=out_channel,out_channels=out_channel,kernel_size=3,stride=1,padding=1,bias=False)
    self.bn2 = nn.BatchNorm2d(out_channel)
    self.downsample = None
    
  def forward(self,x):
    identity = x
    if self.downsample is not None:
      identity = self.downsample(X)
      
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    
    out = self.conv2(out)
    out = self.bn2(out)
    
    out += identity
    out = self.relu(out)
    
    return out
  
class Bottleneck(nn.Module):
  expansion = 4
  
  def __init__(self, in_channel, out_channel, stride = 1, downsample = None):
    super(Bottleneck, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, bias=False)
    self.bn1 = nn.BatchNorm2d(out_channel)
    self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=stride, bias=False, padding=1)
    #因为在这一部分虚线的残差与实线的残差所对应的卷积核的stride不一样，所以选用一个传入参数
    self.bn2 = nn.BatchNorm2d(out_channel)
    self.cn3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*self.expansion, kernel_size=1, stride=1, bias=False)
    self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
    self.relu = nn.ReLU(inplace=True)
    
