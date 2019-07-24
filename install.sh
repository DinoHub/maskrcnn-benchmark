sudo pip3 install -r requirements.txt

export INSTALL_DIR=$PWD

# install pycocotools
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
sudo python3 setup.py build_ext install

# install apex
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
sudo python3 setup.py install --cuda_ext --cpp_ext

# install PyTorch Detection
cd $INSTALL_DIR
# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
sudo python3 setup.py build develop


unset INSTALL_DIR
