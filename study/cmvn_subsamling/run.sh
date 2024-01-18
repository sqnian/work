set -euox pipefail

# if [[ -d build ]]; then
#     rm -rf build/
# fi

# if [[ -d dist ]]; then
#     rm -rf dist/
# fi

# if [[ -d test.egg-info ]]; then
#     rm -rf test.egg-info/
# fi

# python3 setup.py install

bash build.sh

python3 test.py