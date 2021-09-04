pip install -U easynmt
pip install gdown

sudo apt-get install libgoogle-perftools-dev libsparsehash-dev
git clone https://github.com/clab/fast_align.git
mkdir fast_align/build

cd fast_align/build
cmake ..
make
cd ..
cd ..
git clone https://github.com/neulab/awesome-align.git
cd awesome-align/

pip install -r requirements.txt

python3 setup.py install
cd ..