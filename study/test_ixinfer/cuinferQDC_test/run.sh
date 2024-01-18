set -euox pipefail

if [[ -d build ]]; then
    rm -rf build/
fi

if [[ -d dist ]]; then
    rm -rf dist/
fi

if [[ -d infer_test.egg-info ]]; then
    rm -rf infer_test.egg-info/
fi

python3 setup.py install

# python3 gpu_test_signal.py
# python3 gpu_test_double.py

# python3 cpu_test.py



