# python基础语法

## 基本数据类型

### 1.Number（数字类型）

python的数字类型包括int,float,bool,和complex的类型。当指定一个值，就创建了Number的对象


```python
i = 3
print(id(i))
i = i+1;
print(id(i))
```

    2842601744752
    2842601744784


### 2.String（字符串类型）

字符串支持切片访问，字符串的数据使用''或者""括起来。支持正序读取和逆序读取


```python
str = 'Picture'
print(str[1:3])
print(str[-3:-1])
print(str[3:-1])
print(str[2:])
print(2*str)
print(str+"Text")
```

    ic
    ur
    tur
    cture
    PicturePicture
    PictureText



```python
word = 'python'
print(word[0],word[5])
word[0]='Q'#无法修改word字符串，会产生报错
#如果需要修改内容，需要重新赋值
word = 'Quit'
```

    p n


### 3、List（列表）类型

List使用方括号[]进行定义，并且支持切片索引，List元素中的数据类型可以不相同


```python
list = ['a',1,1.13,'Hello',[7,8,9]]
print(list)
print(list[4])
print(list[-2:5])
print(list[2:])
```

    ['a', 1, 1.13, 'Hello', [7, 8, 9]]
    [7, 8, 9]
    ['Hello', [7, 8, 9]]
    [1.13, 'Hello', [7, 8, 9]]


list列表中的单个元素可以进行修改，并且list还内置了许多方法。例如拼接的append，去头部的pop函数


```python
a = [1,2,3,4,5,6,7,8,9]
a[0] = 9
print(a)
a.append(9)
print(a)
a[2:5]=[]
print(a)
a.pop(2)
print(a)
```

    [9, 2, 3, 4, 5, 6, 7, 8, 9]
    [9, 2, 3, 4, 5, 6, 7, 8, 9, 9]
    [9, 2, 6, 7, 8, 9, 9]
    [9, 2, 7, 8, 9, 9]


list常用的遍历方式一共有三种，一种是for循环遍历，一种是ecumerate函数，最后一种是通过下标遍历

### 4、Tuple（元组）类型

元组和列表很相似，但是不同的是元组里面的内容不可以进行修改，使用()进行定义


```python
tuple = ('Spiderman',2017,33.4,'HomeComing',14)
tinytuple = (16,'Marvel')
print(tuple)
print(tuple[1:3])
print(tuple[0])
print(tuple+tinytuple)#元组连接
```

    ('Spiderman', 2017, 33.4, 'HomeComing', 14)
    (2017, 33.4)
    Spiderman
    ('Spiderman', 2017, 33.4, 'HomeComing', 14, 16, 'Marvel')


### 5、Dictionary(字典)

字典是一种特殊的存储结构，里面存储的元素包括了key值和value值两种数据，key值和value值的数据类型可以任意,字典使用{}进行定义


```python
#字典访问
dict = {'Name':'Mary','Age':7,'Class':'First'}
print(dict)
print('Name',dict['Name'])
```

    {'Name': 'Mary', 'Age': 7, 'Class': 'First'}
    Name Mary



```python
#修改字典
dict ={'Name':'Mary','Class':'First'}
#添加
dict['Gender'] = 'Female'
print(dict)
#修改
dict.update({"No":"001"})
print(dict)
#删除
del dict['Gender']
print(dict)
#清除
dict.clear()
print(dict)
```

    {'Name': 'Mary', 'Class': 'First', 'Gender': 'Female'}
    {'Name': 'Mary', 'Class': 'First', 'Gender': 'Female', 'No': '001'}
    {'Name': 'Mary', 'Class': 'First', 'No': '001'}
    {}


### 6、set（集合）

集合大体上与list相似，但是很大的不同是集合中不会出现重复的元素，同时每次输出集合的元素的顺序都有可能不相同


```python
#创建集合，创建集合可以使用set函数或者是{}进行创建
#空集合
var = set()
print(var,type(var))
#有数据的集合
var = {'LiLei','ZhangSan','LiSi','LiLei'}
print(var,type(var))
```

    set() <class 'set'>
    {'LiSi', 'LiLei', 'ZhangSan'} <class 'set'>



```python
#集合成员检测
#判断元素是否在集合内
result = 'LiLei' in var
print(result)
#判断元素不在集合内
result = 'LiLei' not in var#区分大小写
print(result)
```

    True
    False



```python
#添加删除元素
var = {'LiLei','ZhangSan','LiSi'}
var.add('LiBai')#add方法添加
print(var)
var.update('DuFu')#update更新，同时也可以用作添加,拆分元素依次添加
print(var)
var.remove('D')
var.remove('u')
var.remove('F')
print(var)
```

    {'LiSi', 'LiLei', 'ZhangSan', 'LiBai'}
    {'D', 'ZhangSan', 'LiBai', 'LiSi', 'F', 'u', 'LiLei'}
    {'ZhangSan', 'LiBai', 'LiSi', 'LiLei'}



```python
#集合中元素的遍历
anml = {'a','b','c','d'}
for item in anml:
    print(item)

for item in enumerate(anml):
    print(item)
```

    c
    a
    d
    b
    (0, 'c')
    (1, 'a')
    (2, 'd')
    (3, 'b')


## 数据文件读写

### 1、打开文件

python内置了open函数供打开文件，打开后会创建一个文件的对象
语法：open(file_name,[,accesss_mode][,buffering])  
主要参数：file_name:文件名称，字符串类型  
    access_mode:文件打开模式，常用的有'w'和'a'模式，分别为改写和添加  
    buffering：文件缓冲区的策略，可选  
    例如：f = open('file.txt','w')  

### 2、写入文件

向文件内写入数据，可以使用文件对象的write方法，参数为要写入的字符串
例如：f.write('some data')

### 3、关闭文件

关闭文件常用的方法为close方法，例如：f.close()


```python
#打开文件并写入数据
file = 'Info.txt'
f = open(file,'w')
f.write("I am LiHua")
f.close()
#也可以使用with语句打开文件
with open('INFO.txt','a') as f:
    f.write("I am LiHua")
```

### 4、读取文件内容

文件对象提供了读取文件的方法，包括了read(),readlines(),readline()等方法

**1.file.read([count])**  
读文件，默认读取整个文件，如果设置了参数count，就读取count个字节,返回值为字符串  
**2.file.readline()**  
从当前位置，读取文件的一行，返回值为字符串  
**3.file.readlines()**  
从当前位置，读取文件的所有航，返回值的列表


```python
#read函数读取整个文件
with open('INFO.txt') as f:
    ct10 = f.read(5)#读取5个字符
    print(ct10)
    print("======")
    contents = f.read()
    print(contents)
#有些读取的数据需要去掉空格、换行、回车等,可以使用python自带的函数去除
#strcip():去除头、尾的字符和空白符
#lstrip()：去除开头的字符、空白符
#rstrip()：去除结尾的字符、空白符
```

    I am 
    ======
    LiHuaI am LiHua



```python
#使用readline()函数逐行读取
with open('data.txt',encoding = "utf-8") as f:
    line1 = f.readline()
    line2 = f.readline()
    print(line1)
    print(line2)
    print(line1.strip())
    print(line1.split())
    print(line2.strip())
```

    命令模式 按esc键进入命令模式。
    
    在命令行模式在，按m,y切换markdown和code模式 命令模式下，单元格边框为灰色，且左侧边框线为蓝色粗线条。
    命令模式 按esc键进入命令模式。
    ['命令模式', '按esc键进入命令模式。']
    在命令行模式在，按m,y切换markdown和code模式 命令模式下，单元格边框为灰色，且左侧边框线为蓝色粗线条。



```python
#使用readlines一次读取多行
with open('data.txt',encoding = "utf-8") as f:
    lines = f.readlines()
print(lines)
print("==================")
for line in lines:
    print(line.rstrip())
```

    ['命令模式 按esc键进入命令模式。\n', '在命令行模式在，按m,y切换markdown和code模式 命令模式下，单元格边框为灰色，且左侧边框线为蓝色粗线条。']
    ==================
    命令模式 按esc键进入命令模式。
    在命令行模式在，按m,y切换markdown和code模式 命令模式下，单元格边框为灰色，且左侧边框线为蓝色粗线条。



```python
#使用for循环逐行读取文件
with open('data.txt',encoding = "utf-8") as f:
    for linedata in f:
        print(linedata.rstrip())
```

    命令模式 按esc键进入命令模式。
    在命令行模式在，按m,y切换markdown和code模式 命令模式下，单元格边框为灰色，且左侧边框线为蓝色粗线条。


### 5、将数据写入文件


```python
#新建文件并写入内容
filename = "write_data.txt"
with open(filename,'w') as f:
    f.write('I am LiHua \n')
    f.write('I am studing English!')
```


```python
#向文件追加数据
with open(filename,'a') as f:
    f.write('I am a student!')
```

### 6、pandas存取文件
pandas和numpy是一个非常强大的工具，常常用于数据处理，这里简单了解pandas和numpy的数据存取功能

**1.read_csv()函数**  
read_csv是用于读取csv文件，csv文件是表格数据文件  
read_csv的格式如下：  
pd.read_csv(filepath_or_buffer,sep,headwe,encoding,index_col,cloums)  
filepath_or_buffer:文件的存储位置，字符串类型  
sep:数据的分隔符，默认为','，字符串类型  
header:整形或者是整数列表，表示此行为关键字  
encoding：字符串行，可选参数  
index_col:整数，索引的列号  


```python
#read_csv读取有标题的数据
import pandas as pd
data1 = pd.read_csv("data.txt")
print(data1)
data2 = pd.read_csv("data.txt",sep = ' ')
print(data2)
```

                                              命令模式 按esc键进入命令模式。
    在命令行模式在，按m  y切换markdown和code模式 命令模式下，单元格边框为灰色，且左侧边框线为蓝色粗线条。
                                命令模式                  按esc键进入命令模式。
    0  在命令行模式在，按m,y切换markdown和code模式  命令模式下，单元格边框为灰色，且左侧边框线为蓝色粗线条。



```python
#read_table读取无标题的数据
import pandas as pd
data3 = pd.read_table("data.txt",sep = ' ')
print(data3)
print("===========================")
data4 = pd.read_table("data.txt",sep = ' ',header=None)
print(data4)
```

                                命令模式                  按esc键进入命令模式。
    0  在命令行模式在，按m,y切换markdown和code模式  命令模式下，单元格边框为灰色，且左侧边框线为蓝色粗线条。
    ===========================
                                   0                             1
    0                           命令模式                  按esc键进入命令模式。
    1  在命令行模式在，按m,y切换markdown和code模式  命令模式下，单元格边框为灰色，且左侧边框线为蓝色粗线条。


### numpy存取文件
numpy库可以通过loadtxt函数从文本文件中读取数据，通过savetxt将数据写入文件


```python
#使用loadtxt读取文件
import numpy as np
tmp = np.loadtxt('data.txt',dtype=np.str,delimiter=" ",encoding = "utf-8")
print(tmp)
```

    [['命令模式' '按esc键进入命令模式。']
     ['在命令行模式在，按m,y切换markdown和code模式' '命令模式下，单元格边框为灰色，且左侧边框线为蓝色粗线条。']]


    C:\Users\HP\AppData\Local\Temp\ipykernel_11836\1156641432.py:3: DeprecationWarning: `np.str` is a deprecated alias for the builtin `str`. To silence this warning, use `str` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.str_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      tmp = np.loadtxt('data.txt',dtype=np.str,delimiter=" ",encoding = "utf-8")



```python
#使用savetxt读取文件
import numpy as np
x = y = z = np.arange(0,50,4.5)
np.savetxt('x.txt',x,delimiter=" ",fmt='%1.5f')
```

# numpy、matplotlib、pandas的用法

## numpy的基础用法

### 1、ndarray对象
numpy强大的功能是基于底层的一个darray的一个库，可以生成n维数组，也同样可以通过下标和切片访问数据  
创建ndarray可通过ndarray和array两个函数来进行创建，格式如下：  
np.array(object,dtype=None,copy=None,order=None,subok=False,ndmin=0)  
object:数组或嵌套的类型  
dtype:数组元素的类型  
order:创建数组的样式  
ndim:生成数组的最小维度  


```python
#创建ndarray对象
#创建一维数组
import numpy as np
a = np.array([1,2,3])
print(a)
#创建二维数组
a = np.array([[1,2],[3,4]])
print(a)
#使用admin设置数组的最小维度
a = np.array([1,2,3,4,5],ndmin = 2)
print(a)
#使用dtype使得数组的类型为复数
a = np.array([1,2,3],dtype = complex)
print(a)
```

    [1 2 3]
    [[1 2]
     [3 4]]
    [[1 2 3 4 5]]
    [1.+0.j 2.+0.j 3.+0.j]


### 2、numpy数据类型
numpy中有24中数据类型对象，与python相对应，也支持python的数据类型  
numpy中的dtype用于生成数据类型对象，语法如下：  
numpy.dtype = (object,align,copy)  
onject:要转换成dtype的数据对象  
align：如果为True，则填充字段  
copy：指明是否fuzhildtype对象  


```python
#使用dtype对象设置数据类型
import numpy as np
x= np.array(5,dtype="float32")
print("x:",x)
print("x对象的data属性:",x)
print("x对象的size属性:",x)
print("x对象的维数:",x)
y = np.array(x,dtype="bool_")
print("转换为bool类型的x大小为",y)
```

    x: 5.0
    x对象的data属性: 5.0
    x对象的size属性: 5.0
    x对象的维数: 5.0
    转换为bool类型的x大小为 True



```python
#使用astype转换为DataFrame
import pandas as pd
df = pd.DataFrame([{'qty':3,'num':50},{'qty':7,'num':20}])
print(df.dtypes)
print("=========================")
df['qty']=df['qty'].astype("int")
df['num']=df['num'].astype("float32")
```

    qty    int64
    num    int64
    dtype: object
    =========================


### 3、numpy数组属性
(轴)：每个线性数组都有一个轴，轴机数组的维度，如果将一个二维数组堪称一维，此一维数组的每个元素也是一个一维元素。  
则每个一个数组都是numpy中的轴。第一个轴相当于底层数组，第二个轴相当于是底层数组中的数组  
(秩):秩描述numpy数组的维数，即轴的数量  
在使用是axis = 0表示在第0轴的操作，axis = 1表示在第一轴上的操作  


```python
#使用axis参数设置当前轴
import numpy as np
arr = np.array([[0,1,2],[3,4,5]])
print(arr)
print(arr.sum(axis=0))
print(arr.sum(axis=1))
```

    [[0 1 2]
     [3 4 5]]
    [3 5 7]
    [ 3 12]



```python
#使用reshape调整数组大小
import numpy as np
arr = np.array([0,1,2,3,4,5,6,7])
print("秩为",arr.ndim)
arr3d = arr.reshape(2,2,2)
print("秩为",arr3d.ndim)
```

    秩为 1
    秩为 3



```python
#显示数组维度和调整数组大小
import numpy as np
a = np.array([[1,2,3],[4,5,6]])
print(a.shape)
a.shape = (3,2)
print(a)
```

    (2, 3)
    [[1 2]
     [3 4]
     [5 6]]


### 4、其他数组产生方式
numpy中还有很多其他的产生方式，下列将进行描述


```python
#numpy.empty
#创建指定大小的空数组
import numpy as np
x = np.empty([3,2],dtype=int)
print(x)
#numpy.zeros创建全0填充的数组
x = np.zeros((2,2),dtype = float)
print(x)
#numpy.ones创建全1填充的数组
x = np.ones((2,2),dtype = float)
print(x)
```

    [[         0 1072693248]
     [         0 1073741824]
     [         0 1074266112]]
    [[0. 0.]
     [0. 0.]]
    [[1. 1.]
     [1. 1.]]



```python
#range函数
#生成间隔固定的数组，大小为设定的范围内
import numpy as np
x = range(0,5,1)
print(x)
#numpy.arange函数
#从设定的范围内生成间隔固定的数组
x = np.arange(0,10,1)
print(x)
#nunpy.linspace
#从设定的范围内生成间隔固定的数组,最后的属性表示生成的个数
x = np.linspace(1,10,10)
print(x)
```

    range(0, 5)
    [0 1 2 3 4 5 6 7 8 9]
    [ 1.  2.  3.  4.  5.  6.  7.  8.  9. 10.]



```python
#使用随机函数创建随机数组
#生成2行3列的浮点随机数组
x = np.random.rand(2,3)
print(x)
#生成2行3列的10以内的整数数组
x = np.random.randint(0,10,(2,3))
print(x)
#生成2行3列的5以内的浮点随机数组
x = np.random.uniform(1,5,(2,3))
print(x)
```

    [[0.73107632 0.72877926 0.58029651]
     [0.09996418 0.45170658 0.89064288]]
    [[6 1 6]
     [8 0 1]]
    [[2.26452817 1.73768983 4.37940401]
     [2.73205466 4.73382009 3.89459684]]



```python
#其他数据类型转换为mdarray
import numpy as np
#list
data = [[1,2,3],[3,4,5],[5,6,7]]
print(type(data))
#list to ndarray
ndarray = np.array(data)
print(type(ndarray))
```

    <class 'list'>
    <class 'numpy.ndarray'>


### 5、numpy计算
numpy库中有许多计算方式，下面将为你展示


```python
#使用numpy.where实现数据的筛选
import numpy as np
num = np.random.normal(0,1,(3,4))
print(num)
num[num<0.5]=0
print(num)
print(np.where(num>0.5,1,0))
```

    [[-0.24755793  0.46530786 -0.42140381  1.1168427 ]
     [-1.05757497 -0.02906053 -0.34401351 -0.163748  ]
     [-1.46702657  0.79202381  0.84770139 -0.7658521 ]]
    [[0.         0.         0.         1.1168427 ]
     [0.         0.         0.         0.        ]
     [0.         0.79202381 0.84770139 0.        ]]
    [[0 0 0 1]
     [0 0 0 0]
     [0 1 1 0]]



```python
#统计运算
#ndarray的统计运算
import numpy as np
score = np.array([[80,88],[82,81],[84,75],[86,83],[75,81]])
#求每列的最大值
result = np.max(score,axis = 0)
print(result)
#求每行的最大值
result = np.max(score,axis = 1)
print(result)
#求每行的最小值
result = np.min(score,axis = 1)
print(result)
```

    [86 88]
    [88 82 84 86 81]
    [80 81 75 83 75]


## pandas的使用
pandas是一个非常强大的库，主要是为了解决数据分析的问题而存在的，里面提供了强大的数据操作和分析工具，提高处理数据的效率。  
pandas主要处理三种数据：  
1、Series：一维数据类型  
2、DataFrame：二维数据类型  
3、Panel：三维数据类型  

### 1、Series数据结构
Series是一种一维数组的对象，它有一维数据和一维数据的标签构成，数据可以是任何类型的numpy数据


```python
#创建Series对象
#使用列表创建
import pandas as pd
a = pd.Series([1,3,5,7,9])
print(a)
#使用字典创建
b = pd.Series({'Longitude':39,'Latitude':116,'Temperature':23})
print(b['Longitude'])
#使用range函数生成
c = pd.Series([3.4,0.8,2.1,0.3,1.5],range(5,10))
print(c[5])
#访问Series数据
#Series可以通过赋值的操作直接修改其中的值，也同样可以进行批量修改
#修改数据
b['city'] = 'Beijing'
b['Temperature'] += 2
print(b)
#筛选数据
print(c[c>2])
#增加对象成员
stiny = pd.Series({'humidity':84})
d = b.append(stiny)
print(b)
print(d)
#删除数据
b = b.drop('city')
print(b)
```

    0    1
    1    3
    2    5
    3    7
    4    9
    dtype: int64
    39
    3.4
    Longitude           39
    Latitude           116
    Temperature         25
    city           Beijing
    dtype: object
    5    3.4
    7    2.1
    dtype: float64
    Longitude           39
    Latitude           116
    Temperature         25
    city           Beijing
    dtype: object
    Longitude           39
    Latitude           116
    Temperature         25
    city           Beijing
    humidity            84
    dtype: object
    Longitude       39
    Latitude       116
    Temperature     25
    dtype: object


    C:\Users\HP\AppData\Local\Temp\ipykernel_11836\4089205693.py:22: FutureWarning: The series.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      d = b.append(stiny)


### DataFrame对象
DataFrame对象是二维数据包含一组有序数列。列索引对应表格的字段名，行索引对应表格的行号，值为一个二维数组


```python
#创建DataFrame对象
#字典构建DataFrame对象
import pandas as pd
dict1 = {'col1':[1,2,5,7],'col2':['a','b','c','d']}
df = pd.DataFrame(dict1)
print(df)
#数组构建DataFrame
a = pd.DataFrame([[1,1,5],[2,2,6],[4,3,5]],columns = ['t1','t2','t3'])
print(a)
#ndarray创建DataFrame
a = np.array([[1,2,3],[4,5,6],[7,8,9]])
b = pd.DataFrame(a)
print(b)
```

       col1 col2
    0     1    a
    1     2    b
    2     5    c
    3     7    d
       t1  t2  t3
    0   1   1   5
    1   2   2   6
    2   4   3   5
       0  1  2
    0  1  2  3
    1  4  5  6
    2  7  8  9



```python
#访问DataFrame对象
#对Series和DataFrame进行索引
import numpy as np
import pandas as pd
ser = pd.Series(np.arange(4),index = ['A','B','C','D'])
data = pd.DataFrame(np.arange(16).reshape(4,4),index = ['BJ','SH','GZ','SZ'],columns = ['q','r','s','t'])
print(ser['C'])
print(ser[2])
print(data['q'])
print(data[['q','t']])
#数据切片和筛选
print(data[:2])
print(data['s']<10)
#抽取指定行、列的数据
print(data.loc[['SH','GZ'],['r','s']])
print(data.iloc[:-1,1:3])
#修改数据
data['q']['BJ'] = 8
data['t'] = 8
print(data)
#增加一列
data['u'] = 9
print(data)
#删除数据
dt1 = data.drop('SZ',axis = 0)
print(dt1)
dt2 = data.drop(['r','u'],axis = 1)
print(dt2)
data.drop('SZ',inplace = True)
print(data)
#汇总和描述性统计运算
#求和
print(data.sum(axis=1))
#求平均值
print(data.mean(axis=1))
#求最大值和最小值
print(data.max())
print(data.min())
#求方差
print(data.std(axis = 0))
```

    2
    2
    BJ     0
    SH     4
    GZ     8
    SZ    12
    Name: q, dtype: int32
         q   t
    BJ   0   3
    SH   4   7
    GZ   8  11
    SZ  12  15
        q  r  s  t
    BJ  0  1  2  3
    SH  4  5  6  7
    BJ     True
    SH     True
    GZ    False
    SZ    False
    Name: s, dtype: bool
        r   s
    SH  5   6
    GZ  9  10
        r   s
    BJ  1   2
    SH  5   6
    GZ  9  10
         q   r   s  t
    BJ   8   1   2  8
    SH   4   5   6  8
    GZ   8   9  10  8
    SZ  12  13  14  8
         q   r   s  t  u
    BJ   8   1   2  8  9
    SH   4   5   6  8  9
    GZ   8   9  10  8  9
    SZ  12  13  14  8  9
        q  r   s  t  u
    BJ  8  1   2  8  9
    SH  4  5   6  8  9
    GZ  8  9  10  8  9
         q   s  t
    BJ   8   2  8
    SH   4   6  8
    GZ   8  10  8
    SZ  12  14  8
        q  r   s  t  u
    BJ  8  1   2  8  9
    SH  4  5   6  8  9
    GZ  8  9  10  8  9
    BJ    28
    SH    32
    GZ    44
    dtype: int64
    BJ    5.6
    SH    6.4
    GZ    8.8
    dtype: float64
    q     8
    r     9
    s    10
    t     8
    u     9
    dtype: int64
    q    4
    r    1
    s    2
    t    8
    u    9
    dtype: int64
    q    2.309401
    r    4.000000
    s    4.000000
    t    0.000000
    u    0.000000
    dtype: float64


### 数据对其
对于许多的应用来说，需要的是运算中的数据对其，即对其索引大小不同的数据



```python
#算数运算的数据对其 
#Series运算中的数据对齐
ser1 = pd.Series({'color':1,'size':2,'weight':3})
ser2 = pd.Series([5,6,3,24],index=['color','size','weight','price'])
print(ser1+ser2)
#DataFrame运算中数据对齐
data1 = pd.DataFrame(np.arange(16).reshape(4,4),index = ['BJ','SH','GZ','SZ'],columns = ['q','r','s','t'])
data2 = pd.DataFrame(np.arange(4).reshape(2,2),index = ['BJ','SZ'],columns = ['r','t'])
print(data1.add(data2,fill_value = 0))
```

    color     6.0
    price     NaN
    size      8.0
    weight    6.0
    dtype: float64
           q     r     s     t
    BJ   0.0   1.0   2.0   4.0
    GZ   8.0   9.0  10.0  11.0
    SH   4.0   5.0   6.0   7.0
    SZ  12.0  15.0  14.0  18.0



```python
#缺失数据的处理
#使用dropna函数可以过滤空值，dropna函数不修改原有的数据，而是生成新的数组
import pandas as pd
from numpy import nan as NA
dt1 = pd.DataFrame(np.arange(16).reshape(4,4),index = ['BJ','SH','GZ','SZ'],columns = ['q','r','s','t'])
dt2 = pd.DataFrame(np.arange(12).reshape(4,3),index = ['BJ','SH','GZ','SZ'],columns = ['q','r','s'])
df = dt1 + dt2
Hfinedf = df.dropna()
Vfinedf = df.dropna(axis = 1)
print(df)
print(Hfinedf)
print(Vfinedf)
#s使用how过滤DataFrame数组的缺失值
dtHow1 = pd.DataFrame([[0,0,0,0],[0,0,0,0],[NA,0,0,0],[NA,NA,NA,NA]])
dtHow2 = dtHow1.dropna(axis=0,how='all')
print(dtHow2)
#使用notnull函数判断空值
print(df.notnull())
#使用notnull函数过滤空值
s1 = pd.Series(['ONE','TWO',NA,None,'TEN'])
print(s1[s1.notnull()])
```

         q   r   s   t
    BJ   0   2   4 NaN
    SH   7   9  11 NaN
    GZ  14  16  18 NaN
    SZ  21  23  25 NaN
    Empty DataFrame
    Columns: [q, r, s, t]
    Index: []
         q   r   s
    BJ   0   2   4
    SH   7   9  11
    GZ  14  16  18
    SZ  21  23  25
         0    1    2    3
    0  0.0  0.0  0.0  0.0
    1  0.0  0.0  0.0  0.0
    2  NaN  0.0  0.0  0.0
           q     r     s      t
    BJ  True  True  True  False
    SH  True  True  True  False
    GZ  True  True  True  False
    SZ  True  True  True  False
    0    ONE
    1    TWO
    4    TEN
    dtype: object


## Matplotlib
matplotlib是python一个最基本的图形绘图库，提供了多种函数和属性值来让我们画图
下列将介绍matplotlib几个常用的画图函数和属性

**add_subplot函数**
函数使用方法如下：
<子图对象>=<figure对象>.add_subplot(nrows,ncols,index)  
nrows:子图划分成的行数  
ncols:子图划分成的列数  
index:当前子图的序号  


```python
import matplotlib.pyplot as plt
fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax1 = fig.add_subplot(2,2,2)
```


​    
![png](output_83_0.png)
​    



```python
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,3)
```


​    
![png](output_84_0.png)
​    


**plot函数**  
绘制图像曲线可以使用plot函数，绘制需要在画布上进行，因此也需要先创建一个画布对象  
基本格式：  
plt.plot(x,y,format_string,**kwargs)  
x:x轴的数据，一维类型的数据  
y:y轴的数据，一维类型的数据  
format_string:控制曲线的格式字符串  


```python
#绘制简单的直线
import matplotlib.pyplot as plt
import numpy as np
a = np.arange(10)
plt.xlabel('x')
plt.ylabel('y')
plt.plot(a,a*1.5,a,a*2.5,a,a*3.5,a,a*4.5)
plt.legend(['1.5x','2.5x','3.5x','4.5x'])
plt.title('simple lines')
plt.show()
```


​    
![png](output_86_0.png)
​    



```python
#pandas内置的绘图函数
import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import matplotlib.pyplot as plt

s1 = Series(np.random.randn(1000)).cumsum()
s2 = Series(np.random.randn(1000)).cumsum()

plt.subplot(2,1,1)
ax1 = s1.plot(kind = 'line',label = 'S1' ,title ='Series',style='--')
s2.plot(ax=ax1,kind='line',label='S2')
plt.ylabel('value')
plt.legend(loc=2)
plt.subplot(2,1,2)
s1[0:10].plot(kind='bar',grid=True,label='S1')
plt.xlabel('index')
plt.ylabel('value')
```




    Text(0, 0.5, 'value')




​    
![png](output_87_1.png)
​    



```python
#绘制3D图形
#使用Axes3D.scatter函数绘制三维散点图
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def randrange(n,randFloor,randCeil):
    rnd = np.random.rand(n)
    return (randCeil-randFloor)*rnd+randFloor

plt.rcParams['font.sans-serif'] = ['SimHei']#显示中文
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111,projection='3d')
n = 100
for zmin,zmax,c,m,l in [(4,15,'r','o','低值'),(13,20,'g','*','高值')]:
    x = randrange(n,0,20)
    y = randrange(n,0,20)
    z = randrange(n,zmin,zmax)
    ax.scatter(x,y,z,c=c,marker=m,label=l,s=z*6)
    
ax.set_xlabel('X-value')
ax.set_ylabel('Y-value')
ax.set_zlabel('Z-value')
ax.set_title('高低值离散图',alpha = 0.6,size = 15,weight = 'bold')
ax.legend(loc=0)
plt.show()
```


​    
![png](output_88_0.png)
​    

