# 一、前言

## 1.2 关键组件

### 1.2.1 数据

每个**数据集**由⼀个个**样本**组成， ⼤多时候，它们遵循独⽴同分布。样本有时也叫做**数据点**或者**数据实例**，通常每个样本由⼀组称为特征的属性组成。机器学习模型会根据这些属性进⾏预测。在上⾯的监督学习问题中，要预测的是⼀个特殊的属性，它被称为**标签或⽬标**。

当每个样本的特征类别数量都是相同的时候，其特征向量是固定⻓度的，这个⻓度被称为**数据的维数**。固定⻓度的特征向量是⼀个⽅便的属性，它有助于我们量化学习⼤量样本。

我们拥有的数据越多，我们的⼯作就越容易。当我们有了更多的数据，我们通常可以训练出更强⼤的模型，从⽽减少对预先设想假设的依赖。仅仅拥有海量的数据是不够的，我们还需要正确的数据。

### 1.2.2 模型

深度学习与经典⽅法的区别主要在于：前者关注的功能强⼤的模型，这 些模型由神经⽹络错综复杂的交织在⼀起，包含层层数据转换，因此被称为深度学习。

### 1.2.3 目标函数

在机器学习中，我们需要定义模型的优劣程度的度量，这个度量在⼤多数情况是“可优化”的，我们称之为**⽬标函数**。我们通常定义⼀个⽬标函数，并希望优化它到最低点。因为越低越好，所以这些函数有时被称为**损失函数**。但这 只是⼀个惯例，你也可以取⼀个新的函数，优化到它的最⾼点。这两个函数本质上是相同的，只是翻转⼀下符号。

我们通常将可⽤数据集分成两部分：**训练数据集**⽤于拟合模型参数，**测试数据集**⽤于评估拟合的模型。测试性能可能会显著偏离训练性能。当⼀个模型在训练集上表现良好，但不能推 ⼴到测试集时，我们说这个模型是**“过拟合”**的。

### 1.2.4 优化算法

深度学习中，⼤多流⾏的优化算法通常基于⼀种基本⽅法‒梯度下降。简⽽⾔之，在每个步骤中，梯度下降法都会检查每个参数，看看如果你仅对该参数 进⾏少量变动，训练集损失会朝哪个⽅向移动。然后，它在可以减少损失的⽅向上优化参数。

## 1.3 各种机器学习问题

**监督学习**：已知数据和其一一对应的标签，训练一个智能算法，将输入数据映射到标签的过程。监督学习是最常见的学习问题之一，就是人们口中常说的分类问题。比如已知一些图片是猪，一些图片不是猪，那么训练一个算法，当一个新的图片输入算法的时候算法告诉我们这张图片是不是猪。

**无监督学习**：已知数据不知道任何标签，按照一定的偏好，训练一个智能算法，将所有的数据映射到多个不同标签的过程。相对于有监督学习，无监督学习是一类比较困难的问题，所谓的按照一定的偏好，是比如特征空间距离最近，等人们认为属于一类的事物应具有的一些特点。举个例子，猪和鸵鸟混杂在一起，算法会测量高度，发现动物们主要集中在两个高度，一类动物身高一米左右，另一类动物身高半米左右，那么算法按照就近原则，75厘米以上的就是高的那类也就是鸵鸟，矮的那类是第二类也就是猪，当然这里也会出现身材矮小的鸵鸟和身高爆表的猪会被错误的分类。

**强化学习**：智能算法在没有人为指导的情况下，通过不断的试错来提升任务性能的过程。“试错”的意思是还是有一个衡量标准，用棋类游戏举例，我们并不知道棋手下一步棋是对是错，不知道哪步棋是制胜的关键，但是我们知道结果是输还是赢，如果算法这样走最后的结果是胜利，那么算法就学习记忆，如果按照那样走最后输了，那么算法就学习以后不这样走。

**弱监督学习**： 已知数据和其一一对应的弱标签，训练一个智能算法，将输入数据映射到一组更强的标签的过程。**标签的强弱**指的是标签蕴含的信息量的多少，比如相对于分割的标签来说，分类的标签就是弱标签，如果我们知道一幅图，告诉你图上有一只猪，然后需要你把猪在哪里，猪和背景的分界在哪里找出来，那么这就是一个已知若标签，去学习强标签的弱监督学习问题。

**半监督学习** ：已知数据和部分数据一一对应的标签，有一部分数据的标签未知，训练一个智能算法，学习已知标签和未知标签的数据，将输入数据映射到标签的过程。半监督通常是一个数据的标注非常困难，比如说医院的检查结果，医生也需要一段时间来判断健康与否，可能只有几组数据知道是健康还是非健康，其他的只有数据不知道是不是健康。那么通过有监督学习和无监督的结合的半监督学习就在这里发挥作用了。

**多示例学习** ：已知包含多个数据的数据包和数据包的标签，训练智能算法，将数据包映射到标签的过程，在有的问题中也同时给出包内每个数据的标签。多事例学习引入了数据包的概念，比如说一段视频由很多张图组成，假如1000张，那么我们要判断视频里是否有猪出现，一张一张的标注每一帧是否有猪太耗时，所以人们看一遍说这个视频里有猪或者没猪，那么就得到了多示例学习的数据，1000帧的数据不是每一个都有猪出现，只要有一帧有猪，那么我们就认为这个包是有猪的，所有的都没有猪，才是没有猪的，从这里面学习哪一段视频（1000张）有猪哪一段视频没有就是多事例学习的问题。

**迁移学习**：可以使我们在他人训练过的模型基础上进行小改动便可投入使用。

***

# 二、预备知识

## 2.1 数据操作

**张量**：表⽰由⼀个数值组成的数组，这个数组可能有多个维度。具有⼀个轴的张量对应数学上的向量； 具有两个轴的张量对应数学上的矩阵；具有两个轴以上的张量没有特殊的数学名称。张量中的每个值都称为张量的元素。

```python
# 1.入门.py
import torch

# arange 创建⼀个⾏向量 x
x = torch.arange(12)

print(x)
# 可以通过张量的shape属性来访问张量（沿每个轴的⻓度）的形状。
print(x.shape)
# 张量中元素的总数
print(x.numel())

# 改变⼀个张量的形状⽽不改变元素数量和元素值，可以调⽤reshape函数
# 可以通过-1来调⽤此⾃动计算出维度的功能。即我们可以⽤x.reshape(-1,4)或x.reshape(3,-1)来取代x.reshape(3,4)。
X = x.reshape(3, 4)
print(X)

# 创建⼀个形状为（2,3,4）的张量，其中所有元素都设置为0，2页3行4列
print(torch.zeros((2, 3, 4)))

# 可以创建⼀个形状为(2,3,4)的张量，其中所有元素都设置为1
print(torch.ones((2, 3, 4)))

# 创建⼀个形状为（3,4）的张量。其中的每个元素都从均值为0、标准差为1的标准⾼斯分布（正态分布）中随机采样。
print(torch.randn(3, 4))

# 可以通过提供包含数值的Python列表（或嵌套列表），来为所需张量中的每个元素赋予确定值。在这⾥，最外层的列表对应于轴0，内层的列表对应于轴1。
print(torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]]))
```

```python
# 2.运算符.py
import torch

x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print(x + y)
print(x - y)
print(x * y)
print(x / y)
print(x ** y) # **运算符是求幂运算

# “按元素”⽅式可以应⽤更多的计算，包括像求幂这样的⼀元运算符。 e的几次方
print(torch.exp(x))

# 把多个张量连结在⼀起，把它们端对端地叠起来形成⼀个更⼤的张量。沿⾏（轴-0，形状的第⼀个元素）和按列（轴-1，形状的第⼆个元素）连结两个矩阵
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(torch.cat((X, Y), dim=0))
print(torch.cat((X, Y), dim=1))
print(X == Y)
print(X.sum())
```

```python
# 3.广播机制.py
import torch

a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
print(a)
print(b)
print(a + b)
```

```python
# 4.索引和切片
import torch

X = torch.arange(12, dtype=torch.float32).reshape((3,4))
print(X[-1])
print(X[1:3])

X[1, 2] = 9
print(X)

# 给第一行第二行全部赋值为12
X[0:2, :] = 12
print(X)
```

```python
# 5.节省内存
import torch

X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
before = id(Y)
Y = Y + X
print(id(Y) == before)

# 我们不想总是不必要地分配内存。在机器学习中，我们可能有数百兆的参数，并且在⼀秒内多次更新所有参数。通常情况下，我们希望原地执⾏这些更新。其次，如果我们不原地更新，其他引⽤仍然会指向旧的内存位置，这样我们的某些代码可能会⽆意中引⽤旧的参数。
# 执⾏原地操作⾮常简单，我们可以使⽤切⽚表⽰法将操作的结果分配给先前分配的数组
Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))

# 如果在后续计算中没有重复使⽤X，我们也可以使⽤X[:] = X + Y或X += Y来减少操作的内存开销。
before = id(X)
X += Y
print(id(X) == before)
```

```python
# 6.转换为其他python对象
import torch

X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

# 将深度学习框架定义的张量转换为NumPy张量很容易，反之也同样容易。torch张量和numpy数组将共享它们的底层内存，就地操作更改⼀个张量也会同时更改另⼀个张量。
A = X.numpy()
B = torch.tensor(A)
print(type(A))
print(type(B))

# 要将⼤⼩为1的张量转换为Python标量，我们可以调⽤item函数或Python的内置函数。
a = torch.tensor([3.5])
print(a, a.item(), float(a), int(a))
```

## 2.2 数据预处理

```python
# 1.读取数据集
import os
import pandas as pd

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')

with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n') # 列名
    f.write('NA,Pave,127500\n') # 每⾏表⽰⼀个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

# 加载原始数据集
data = pd.read_csv(data_file)
print(data)
```

```python
# 2.处理缺失值
import os
import pandas as pd

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')

with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n') # 列名
    f.write('NA,Pave,127500\n') # 每⾏表⽰⼀个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

data = pd.read_csv(data_file)

# “NaN”项代表缺失值。为了处理缺失的数据，典型的⽅法包括插值法和删除法，其中插值法⽤⼀个替代值弥补缺失值，⽽删除法则直接忽略缺失值。在这⾥，我们将考虑插值法。通过位置索引iloc，我们将data分成inputs和outputs，其中前者为data的前两列，⽽后者为data的最后⼀列。对于inputs中缺少的数值，我们⽤同⼀列的均值替换“NaN”项。
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())
print(inputs)

# 对于inputs中的类别值或离散值，我们将“NaN”视为⼀个类别。由于“巷⼦类型”（“Alley”）列只接受两种类型的类别值“Pave”和“NaN”，pandas可以⾃动将此列转换为两列“Alley_Pave”和“Alley_nan”。巷⼦类型为“Pave”的⾏会将“Alley_Pave”的值设置为1，“Alley_nan”的值设置为0。缺少巷⼦类型的⾏会将“Alley_Pave”和“Alley_nan”分别设置为0和1。
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
```

```python
# 3.转换为张量格式
import os
import torch
import pandas as pd

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')

with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n') # 列名
    f.write('NA,Pave,127500\n') # 每⾏表⽰⼀个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

data = pd.read_csv(data_file)

inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())

inputs = pd.get_dummies(inputs, dummy_na=True)

# 现在inputs和outputs中的所有条⽬都是数值类型，它们可以转换为张量格式。当数据采⽤张量格式后，可以通过在 2.1节中引⼊的那些张量函数来进⼀步操作。
X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)

print(X)
print(y)
```

## 2.3 线性代数

```python
# 1.标量
import torch

x = torch.tensor(3.0)
y = torch.tensor(2.0)
print(x + y)
print(x * y)
print(x / y)
print(x ** y)
```

```python
# 2.向量
import torch

x = torch.arange(4)
print(x)
print(x[3])
print(len(x))
print(x.shape)
```

```python
# 3.矩阵
import torch

A = torch.arange(20).reshape(5, 4)
print(A)
print(A.T)

B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
print(B)
print(B == B.T)
```

```python
# 4.张量
import torch

X = torch.arange(24).reshape(2, 3, 4)
print(X)
```

```python
# 5.张量算法的基本性质
import torch

A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone() # 通过分配新内存，将A的⼀个副本分配给B
print(A)
print(A + B)
print(A * B)

a = 2
X = torch.arange(24).reshape(2, 3, 4)
print(a + X)
print((a * X).shape)
```

```python
# 6.降维
import torch

x = torch.arange(4, dtype=torch.float32)
print(x)
print(x.sum())

A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
print(A.shape)
print(A.sum())

# 沿轴0相加
A_sum_axis0 = A.sum(axis=0)
print(A_sum_axis0)
print(A_sum_axis0.shape)

# 沿轴1相加
A_sum_axis1 = A.sum(axis=1)
print(A_sum_axis1)
print(A_sum_axis1.shape)

# 沿两个轴相加
print(A.sum(axis=[0, 1]))

# 求平均
print(A.mean())
print(A.sum() / A.numel())

# 沿轴0求平均
print(A.mean(axis=0))
print(A.sum(axis=0) / A.shape[0])

# 非降维求和
sum_A = A.sum(axis=1, keepdims=True)
print(sum_A)
print(A / sum_A)

# 沿某个轴计算A元素的累积总和
print(A.cumsum(axis=0))
```

```python
# 7.点积
import torch

x = torch.arange(4, dtype=torch.float32)
y = torch.ones(4, dtype=torch.float32)
print(x, y, torch.dot(x, y))
print(torch.sum(x * y))
```

```python
# 8.矩阵-向量积
import torch

x = torch.arange(4, dtype=torch.float32)
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
print(A.shape)
print(x.shape)
print(torch.mv(A, x))
```

```python
# 9.矩阵-矩阵乘法
import torch

A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = torch.ones(4, 3)
print(torch.mm(A, B))
```

```python
# 10.范数
import torch

u = torch.tensor([3.0, -4.0])
# 向量的L2范数。
print(torch.norm(u))
# 向量的L1范数。
print(torch.abs(u).sum())
# Frobenius范数满⾜向量范数的所有性质，它就像是矩阵形向量的L2范数。调⽤以下函数将计算矩阵的Frobenius范数。
print(torch.norm(torch.ones((4, 9))))
```

## 2.4 微积分

```python
import numpy as np
from matplotlib_inline import backend_inline
from d2l import torch as d2l


def f(x):
    return 3 * x ** 2 - 4 * x


# 求导函数
def numerical_lim(f, x, h):
    return (f(x + h) - f(x)) / h


h = 0.1
for i in range(5):
    print(f'h={h:.5f}, numerical limit={numerical_lim(f, 1, h):.5f}')
    h *= 0.1


# 注释#@save是⼀个特殊的标记，会将对应的函数、类或语句保存在d2l包中。因此，以后⽆须重新定义就可以直接调⽤它们，（例如，d2l.use_svg_display()）。
def use_svg_display():  #@save
    # """使⽤svg格式在Jupyter中显⽰绘图"""
    backend_inline.set_matplotlib_formats('svg')


# 设置图表⼤⼩
def set_figsize(figsize=(3.5, 2.5)):    #@save
    # """设置matplotlib的图表⼤⼩"""
    use_svg_display()
    d2l.plt.rcParams['figure.figsize'] = figsize


# 设置由matplotlib⽣成图表的轴的属性。
#@save
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    # """设置matplotlib的轴"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()



#@save
def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
    ylim=None, xscale='linear', yscale='linear',
    fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """绘制数据点"""
    if legend is None:
        legend = []
    set_figsize(figsize)
    axes = axes if axes else d2l.plt.gca()
    # 如果X有⼀个轴，输出True
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list) and not hasattr(X[0], "__len__"))
    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)


x = np.arange(0, 3, 0.1)
plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
```

## 2.5 自动微分

```python
# 1.例子.py
import torch

x = torch.arange(4.0)
print(x)

x.requires_grad_(True) # 等价于x=torch.arange(4.0,requires_grad=True)
# 使用x.grad保存梯度, 不会在每次对⼀个参数求导时都分配新的内存, ⼀个标量函数关于向量x的梯度是向量，并且与x具有相同的形状。
print(x.grad) # 默认值是None

y = 2 * torch.dot(x, x)
print(y)

y.backward()
print(x.grad)

x.grad == 4 * x
print(x.grad)

# 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
x.grad.zero_()
y = x.sum()
print(y)
y.backward()
print(x.grad)
```

```python
# 2.非标量变量的反向传播.py
import torch

x = torch.arange(4.0)
x.requires_grad_(True) # 等价于x=torch.arange(4.0,requires_grad=True)

y = x * x
# 等价于y.backward(torch.ones(len(x)))
y.sum().backward()
print(x.grad)
```

```python
# 3.分离计算.py
import torch

x = torch.arange(4.0)

x.requires_grad_(True) # 等价于x=torch.arange(4.0,requires_grad=True)

# x.grad.zero_()
y = x * x
u = y.detach()
z = u * x
z.sum().backward()
print(u, x.grad)
print(x.grad == u)

x.grad.zero_()
y.sum().backward()
print(x.grad == 2 * x)
```

```python
# 4.python控制流的梯度计算
import torch

def f(a):
    b = a * 2
    # print(b.norm())
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(size=(), requires_grad=True)
# print(a)
d = f(a)
d.backward()
# print(a.grad)
print(a.grad == d / a)
```

## 2.6 概率

```python
# 1.基本概率论
import torch
from torch.distributions import multinomial
from d2l import torch as d2l

fair_probs = torch.ones([6]) / 6
print(multinomial.Multinomial(1, fair_probs).sample())

print(multinomial.Multinomial(10, fair_probs).sample())

# 将结果存储为32位浮点数以进⾏除法
counts = multinomial.Multinomial(1000, fair_probs).sample()
print(counts / 1000) # 相对频率作为估计值

counts = multinomial.Multinomial(10, fair_probs).sample((500,))
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)
d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].numpy(), label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend()
```

```python
# 2.查找模块中的所有函数和类
import torch

print(dir(torch.distributions))
```

***

# 三、线性神经网络

## 3.1 线性回归

```python
# 1.矢量化加速
import math
import time
import numpy as np
import torch
from d2l import torch as d2l

n = 10000
a = torch.ones(n)
b = torch.ones(n)

class Timer: #@save
    """记录多次运⾏时间"""
    def __init__(self):
        self.times = []
        self.start()
    def start(self):
        """启动计时器"""
        self.tik = time.time()
    def stop(self):
        """停⽌计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]
    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)
    def sum(self):
        """返回时间总和"""
        return sum(self.times)
    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()

c = torch.zeros(n)
timer = Timer()
for i in range(n):
    c[i] = a[i] + b[i]
print(f'{timer.stop():.5f} sec')

timer.start()
d = a + b
print(f'{timer.stop():.5f} sec')
```

```python
# 2.正太分布与平方损失
import math
import time
import numpy as np
import torch
from d2l import torch as d2l

def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    return p * np.exp(-0.5 / sigma**2 * (x - mu)**2)

# 再次使⽤numpy进⾏可视化
x = np.arange(-7, 7, 0.01)
# 均值和标准差对
params = [(0, 1), (0, 2), (3, 1)]
d2l.plot(x, [normal(x, mu, sigma) for mu, sigma in params], xlabel='x', ylabel='p(x)', figsize=(4.5, 2.5), legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])
```

## 3.2 线性回归的从零开始实现

```python
# 1.生成数据集
import random
import torch
from d2l import torch as d2l

def synthetic_data(w, b, num_examples): #@save
    """⽣成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

print('features:', features[0],'\nlabel:', labels[0])

d2l.set_figsize()
d2l.plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1);
```

```python
# 2.读取数据集
import random
import torch
from d2l import torch as d2l

def synthetic_data(w, b, num_examples): #@save
    """⽣成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

print('features:', features[0],'\nlabel:', labels[0])

d2l.set_figsize()
d2l.plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1);

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break
```

```python
# 3.初始化模型参数
import torch


w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
```

```python
# 4.定义模型
import torch


def linreg(X, w, b): #@save
    """线性回归模型"""
    return torch.matmul(X, w) + b
```

```python
# 5.定义损失函数
def squared_loss(y_hat, y): #@save
    """均⽅损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
```

```python
# 6.定义优化算法
import torch


def sgd(params, lr, batch_size): #@save
    """⼩批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
```

```python
# 7.训练
import random
import torch
from d2l import torch as d2l


def synthetic_data(w, b, num_examples): #@save
    """⽣成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

print('features:', features[0],'\nlabel:', labels[0])

d2l.set_figsize()
d2l.plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1);

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break



w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)


def linreg(X, w, b): #@save
    """线性回归模型"""
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y): #@save
    """均⽅损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def sgd(params, lr, batch_size): #@save
    """⼩批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y) # X和y的⼩批量损失
        # 因为l形状是(batch_size,1)，⽽不是⼀个标量。l中的所有元素被加到⼀起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size) # 使⽤参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')
```

## 3.3 线性回归的简洁实现

```python
import numpy as np
import torch
from torch import nn
from torch.utils import data
from d2l import torch as d2l

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

# 读取数据集
def load_array(data_arrays, batch_size, is_train=True): #@save
    """构造⼀个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


batch_size = 10
data_iter = load_array((features, labels), batch_size)

print(next(iter(data_iter)))

# 3.定义模型
# nn是神经⽹络的缩写
net = nn.Sequential(nn.Linear(2, 1))

# 4.初始化模型参数
net[0].weight.data.normal_(0, 0.01)
print(net[0].bias.data.fill_(0))

# 5.定义损失函数
loss = nn.MSELoss()

# 6.定义优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# 7.训练
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')


w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)
```

## 3.5 图像分类数据集

```python
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display()

# 1.读取数据集
# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
# 并除以255使得所有像素的数值均在0到1之间
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False, transform=trans, download=True)
print(len(mnist_train), len(mnist_test))
print(mnist_train[0][0].shape)

def get_fashion_mnist_labels(labels): #@save
    """返回Fashion-MNIST数据集的⽂本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5): #@save
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图⽚张量
            ax.imshow(img.numpy())
        else:
            # PIL图⽚
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))

# 2.读取⼩批量
batch_size = 256
def get_dataloader_workers(): #@save
    """使⽤4个进程来读取数据"""
    return 4
train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers())
timer = d2l.Timer()
for X, y in train_iter:
    continue
print(f'{timer.stop():.2f} sec')

# 3.整合所有组件
def load_data_fashion_mnist(batch_size, resize=None): #@save
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers()), data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=get_dataloader_workers()))

train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break

print(torch.Size([32, 1, 64, 64]), torch.float32, torch.Size([32]), torch.int64)
```

## 3.6 softmax回归的从零开始实现

```python
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display()

# 1.读取数据集
# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
# 并除以255使得所有像素的数值均在0到1之间
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False, transform=trans, download=True)
print(len(mnist_train), len(mnist_test))
print(mnist_train[0][0].shape)

def get_fashion_mnist_labels(labels): #@save
    """返回Fashion-MNIST数据集的⽂本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5): #@save
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图⽚张量
            ax.imshow(img.numpy())
        else:
            # PIL图⽚
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))

# 2.读取⼩批量
batch_size = 256
def get_dataloader_workers(): #@save
    """使⽤4个进程来读取数据"""
    return 4
train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers())
timer = d2l.Timer()
for X, y in train_iter:
    continue
print(f'{timer.stop():.2f} sec')

# 3.整合所有组件
def load_data_fashion_mnist(batch_size, resize=None): #@save
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers()), data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=get_dataloader_workers()))

train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break

print(torch.Size([32, 1, 64, 64]), torch.float32, torch.Size([32]), torch.int64)
```

## 3.7 softmax回归的简洁实现

```python
import torch
from torch import nn
from d2l import torch as d2l


batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 1.初始化模型参数
# PyTorch不会隐式地调整输⼊的形状。因此，
# 我们在线性层前定义了展平层（flatten），来调整⽹络输⼊的形状
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
net.apply(init_weights);

# 2.重新审视Softmax的实现
loss = nn.CrossEntropyLoss(reduction='none')

# 3.优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

# 4.训练
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

***

# 四、多层感知机