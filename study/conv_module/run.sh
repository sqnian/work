set -euxo pipefail

if [ -d build ]; then
    rm -rf build/
fi

if [ -d dist ]; then
    rm -rf dist/
fi

if [ -d infer_test.egg-info ]; then
    rm -rf infer_test.egg-info/
fi

python3 setup.py install

python3 pointwise_test.py

# python3 depthwise_test.py
# python3 test_glu.py

# python3 test_all.py

