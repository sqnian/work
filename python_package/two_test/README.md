# 源码包安装

## 1.准备工作
```
# 1.首先创建我们需要的目录结构和文件(自行创建)
# 当前测试的目录是： /tmp/demo

`-- two_test
    |-- helloapp
    |   |-- hello.py
    |   `-- __init__.py
    |-- __init__.py
    |-- myapp
    |   |-- __init__.py
    |   `-- myapp.py
    `-- setup.py

# 2.编辑 setup.py
from setuptools import setup, find_packages

setup(
    name="demo",
    version="1.0",
    author="zbj",
    author_email="22@qq.com",
    packages=find_packages(),
)

# 3.编辑 hello.py
def hello_func():
    print("HelloWorld")

# 4.编辑 myapp.py
def myapp_func():
    print("嘿嘿嘿")
```

## 2.源码安装
```
# 进入setup.py所在的那层目录
cd two_test/

# 检查setup.py 是否有错误(warning不是错误)
python setup.py check

# 安装
python setup.py install

#当前环境下所需要的所有依赖包，都保存到 equirements.txt 文件中
pip freeze > requirements.txt
```

## 3.结果

打包之后多出两个文件夹，分别是demo.egg-info和dist。

demo.egg-info是必要的安装信息，

而dist中的压缩包就是安装包，

此时默认的egg包，egg包就是zip包，如果需要使用egg包，name将egg后缀改成zip解压即可

## 4.测试

测试的时候需要注意导包路径和当前所在路径

目前所在路径是: two_test

直接进入python解释器: python3(我自己安装的python3版本)

```
root@software-h3c-r5300-g3-004:/home/shengquan.nian/quan/package/two_test# python3
Python 3.8.12 | packaged by conda-forge | (default, Jan 30 2022, 23:42:07) 
[GCC 9.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from helloapp import hello
>>> hello.hello_func()
HelloWorld
>>> from myapp import myapp
>>> myapp.myapp_func()
嘿嘿嘿
>>> 
```

## 参考链接
https://www.cnblogs.com/Zzbj/p/11535625.html