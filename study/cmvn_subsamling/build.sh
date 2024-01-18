set -euox pipefail

rm -rf build
rm -rf *.so

python3 setup.py build

cp build/lib*/*.so .