# create a new conda environment
conda create --name liga python=3.7 pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
conda activate liga
conda install cmake  # required by spconv

cd ..
## install mmdetection
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/xy-guo/mmdetection_kitti.git
cd mmdetection_kitti
python setup.py develop
cd ..

## install spconv (require boost, cudnn)
git clone https://github.com/traveller59/spconv.git
cd spconv
git reset --hard f22dd9a
git submodule init
git submodule update --recursive
python setup.py bdist_wheel
pip install ./dist/spconv-1.2.1-cp37-cp37m-linux_x86_64.whl
cd ..

cd liga_stereo_det
# install required packages by liga
pip install -r requirements.txt
# install liga in debug mode
python setup.py develop
