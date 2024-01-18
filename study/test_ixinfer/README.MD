

## 单元测试
### 虎哥纯C++ 测试 cuinferQDConvolutionForward api
验证cpu 计算和GPU api计算结果
```
git clone http://bitbucket.iluvatar.ai:7990/scm/~shengquan.nian/study.git
git checkout conv
```



```
# build gtest
cd test_ixinfer
git clone https://github.com/google/googletest.git
cd googletest
mkdir build && cd build
cmake .. -DBUILD_GMOCK=OFF
make -j1
# 切换到test_ixinfert目录下
cd ../../
```


```
# build test_ixinfer

mkdir build && cd build
cmake ..
make
export LD_LIBRARY_PATH=/opt/sw_home/ixinfer/build/lib:$LD_LIBRARY_PATH
./test_qdconv 
```

### python and c++

```
# 切换到test_ixinfer目录下
cd cuinferQDC_test
# compile
bash run.sh

# run CPU compute
python3 cpu_test.py

# run test signal conv + relu  
python3 gpu_test_signal.py

# run test double conv + relu  
python3 gpu_test_double.py
```


