#ResNet可以分成多种亚结构 Res18 Res34 Res50 Res101 Res152

classs BasicBlock (nn.Module):
  expansion = 1
  #expansion这个对应在残差结构中我们主分支的卷积核的个数是否发生变化
  
  def __init__(self, in_channel, out_channel, stride = 1, downsample = None):
    #downsample这个结构对应的是虚线所对应的残差结构，输入和输出的尺寸不一致时需要对输入进行处理
    
