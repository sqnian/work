# 打包上传到pypi

## 1.创建文件的目录结构

```
there_test/
|-- hello
|   |-- hello.py
|   `-- __init__.py
|-- LICENSE
|-- README.md
`-- setup.py
```

## 2. setup.py
```
from setuptools import setup, find_packages

setup(
    name="hello",
    version='1.0',
    description="Test Hello",
    url="None",
    author="nsq",
    author_email="1101@qq.com",
    license="MIT",
    packages=find_packages()
)
```

## 3. LICENSE

LICENSE代表许可证    
https://packaging.python.org/en/latest/tutorials/packaging-projects/

```
Copyright (c) 2018 The Python Packaging Authority

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 4. setuptools 和wheel

首先需要保证你有最新版的setuptools 和wheel

```
python -m pip install --user --upgrade setuptools wheel
```

## 5. 打包模块

```
#当前环境下所需要的所有依赖包，都保存到 equirements.txt 文件中
pip freeze > requirements.txt

python setup.py sdist bdist_wheel
```
打包之后多出两个文件夹，分别是hello.egg-info和dist。hello.egg-info是必要的安装信息，而dist中的压缩包就是安装包

dist中包含两个文件：
```
dist/
|-- hello-1.0-py3-none-any.whl
`-- hello-1.0.tar.gz
````

## 6. 打包方式介绍

有了上面的 setup.py 文件，我们就可以打出各种安装包，   
主要分为两类：sdist 和 bdist。


## 参考链接
https://www.cnblogs.com/Zzbj/p/11535625.html