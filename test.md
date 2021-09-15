## test the NKL dataset with pretrained model
THe results on NKL dataset can be reproduced with following commands:
```
git clone https://github.com/Hanqer/deep-hough-transform.git
cd deep-hough-transform/
wget http://data.kaizhao.net/projects/deep-hough-transform/NKL.zip
unzip NKL.zip -d data/
wget http://data.kaizhao.net/projects/deep-hough-transform/dht_r50_nkl_d97b97138.pth
cd model/_cdht
python setup.py install --user
cd ../..
python forward.py --model dht_r50_nkl_d97b97138.pth --tmp results
python test_nkl.py --pred results/visualize_test/ --gt data/NKL/
```
