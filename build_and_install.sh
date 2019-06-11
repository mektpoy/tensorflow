pip uninstall tensorflow -y
bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package --disk_cache=~/cache
./bazel-bin/tensorflow/tools/pip_package/build_pip_package ~/whl
pip install ~/whl/tensorflow-1.12.0-cp27-cp27mu-linux_x86_64.whl --user
