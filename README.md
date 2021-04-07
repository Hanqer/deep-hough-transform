# Deep Hough Transform

Code accompanying the paper "Deep Hough Transform for Semantic Line Detection" (ECCV2020).
[arXiv2003.04676](https://arxiv.org/abs/2003.04676) | [Online Demo](http://mc.nankai.edu.cn/dht) | [Project page](http://mmcheng.net/dhtline) | [New dataset](http://kaizhao.net/nkl) | [Line Annotator](https://github.com/Hanqer/lines-manual-labeling)

* Training code is open available new.

### Deep Hough Transform
![pipeline](./pipeline.png)

### Requirements
``` 
numpy
scipy
opencv-python
scikit-image
pytorch>=1.0
torchvision
tqdm
yml
deep-hough
```

To install deep-hough, run the following commands.
```sh
cd deep-hough-transform
cd model/_cdht
python setup.py build 
python setup.py install --user
```
Pretrain model (based on ResNet50-FPN): <http://data.kaizhao.net/projects/deep-hough-transform/dht_r50_fpn_sel-c9a29d40.pth>

### Prepare training data
Download original SEL dataset from [here](https://mcl.korea.ac.kr/research/Submitted/jtlee_slnet/ICCV2017_JTLEE_dataset.7z) and extract to `data/` directory. After that, the directory structure should be like:
```
data
├── ICCV2017_JTLEE_gtlines_all
├── ICCV2017_JTLEE_gt_pri_lines_for_test
├── ICCV2017_JTLEE_images
├── prepare_data_JTLEE.py
├── Readme.txt
├── test_idx_1716.txt
└── train_idx_1716.txt
```

Then run python script to generate parametric space label.
```sh
cd deep-hough-tranfrom
python data/prepare_data_JTLEE.py --root './data/ICCV2017_JTLEE_images/' --label './data/ICCV2017_JTLEE_gtlines_all' --save-dir './data/training/JTLEE_resize_100_100/' --list './data/training/JTLEE.lst' --prefix 'JTLEE_resize_100_100' --fixsize 400 --numangle 100 --numrho 100
```

### Training
Following the default config file 'config.yml', you can arbitrarily modify hyperparameters.
Then, run the following command.
```sh
python train.py
```

### Forward
Generate visualization results and save coordinates to _.npy file.
```sh
CUDA_VISIBLE_DEVICES=0 python forward.py --model （your_best_model.pth） --tmp (your_result_save_dir)
```

### Evaluate
Test the EA-score on SEL dataset. After forwarding the model and get the coordinates files. Run the following command to produce EA-score.
```sh
python test.py --pred result/debug/visualize_test/(change to your onw path which includes _.npy files) --gt gt_path/include_txt
```

### License
Our source code is free for non-commercial usage. Please contact us if you want to use it for comercial usage.

