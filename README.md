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



