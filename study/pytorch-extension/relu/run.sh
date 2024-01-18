set -euox pipefail

if [ -d build ];then
    rm -rf build/
fi

if [ -d Conv_relu.egg-info ];then
    rm -rf Conv_relu.egg-info/
fi

if [ -d dist ];then
    rm -rf dist/
fi

python3 setup.py install 

python3 test.py