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
    self.downsample = downsample
    
  def forward(self, x):
    identity = x
    if self.downsample is not None:
      identity = self.downsample(x)
      
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    
    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)
    
    out = self.conv3(out)
    out = self.bn3(out)
    
    return out
  
class ResNet(nn.Module):
  
  def __init__(self, block, blocks_num, num_classes = 1000, include_top = True):
    #这里的block是上面我们定义的残差结构，18和34用的是basicblock, 50等用的就是bottlenet
    #block_num指定的是我们用的残差模块的个数
    #这里的一千是因为Resnet在这里参加的是imagenet，这里是一个一千分类的问题，可以改成你的分类问题的种类
    #include_top是问了方便在以后对残差网络进行升级设置的
    super(ResNet, self).__init__()
    self.include_top = include_top
    self.in_channel = 64
    
    self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
    self.bn1 = nn.BatchNorm2d(self.in_channel)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1 = self._make_layer(block, 64, blocks_num[0])
    self.layer2 = self._make_layer(block, 128, blocks_num[1], stride = 2)
    self.layer3 = self._make_layer(block, 256, blocks_num[2], stride = 2)
    self.layer3 = self._make_layer(block, 256, blocks_num[2], stride = 2)
    if self.include_top:
      self.avepool = nn.AdaptiveAvePool2d((1,1))
      #自适应平均池化，无论输入的特征矩阵的高和宽是什么，得到的特征矩阵的高和宽都是1
      self.fc = nn.Linear(512 * block.expansion, num_classes)
      
    for m in modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=
    
    
  def _make_layer(self, block, channel, block_num, stride = 1):
    #block就是在最上面我们定义的两种block结构
    #channel对应残差结构中某一层中第一个卷积的卷积核个数，比如第一层中Res18和Res34就是2，Res50等就是3
    #block_num表示该层中包含多少个残差结构
    downsample = None
    if stride != 1 or self.in_channel != channel = channel * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False)
        nn.BatchNorm2d(channel * block.expansion))
      
    layers = []
    layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
    self.in_channel = channel * block.expansion
    
    for _ in range(1, block_num):
      layers.append(block(self.in_channel, channel))
      
    return nn.Sequential(*layers)
                                
  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
                                
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)                            
    x = self.layer4(x)                            
                                
    if self.include_top:
      x = self.avgpool(x)
      x = torch.flatten(x, 1)
      x = self.fc(x)
                                
    return x
                                
                                                                
def resnet34(num_classes = 1000, include_top = True):
                                
