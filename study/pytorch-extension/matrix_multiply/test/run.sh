set -euox pipefail

if [ -d build ];then
    rm -rf build/
fi

if [ -d gemm_test.egg-info ];then
    rm -rf gemm_test.egg-info/
fi

if [ -d dist ];then
    rm -rf dist/
fi

python3 setup.py install 

python3 test.py