set -euox pipefail


# test module function
# if [[ -d build ]]; then
#     rm -rf build/
# fi

# if [[ -d dist ]]; then
#     rm -rf dist/
# fi

# if [[ -d infer_test.egg-info ]]; then
#     rm -rf infer_test.egg-info/
# fi

# python3 setup.py install

# test module compile
rm -rf build
rm -rf *.so

python3 setup.py build

cp build/lib*/*.so .

python3 test.py