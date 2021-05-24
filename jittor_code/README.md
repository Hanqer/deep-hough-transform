<h1 style="align: center; color: #159957">Deep Hough Transform for Semantic Line Detection</h1>

High speed performance Jittor implementation Code accompanying the paper "Deep Hough Transform for Semantic Line Detection" (ECCV 2020, PAMI 2021).
[arXiv2003.04676](https://arxiv.org/abs/2003.04676) | [Online Demo](http://mc.nankai.edu.cn/dht) | [Project page](http://mmcheng.net/dhtline) | [New dataset](http://kaizhao.net/nkl) | [Line Annotator](https://github.com/Hanqer/lines-manual-labeling)

* Jittor inference code is open available now. 
* Training code will come soon.

### High speed of Jittor framework
Network inference FPS and speedup ratio (without post processing): 
<table>
   <tr>
      <td rowspan="2"></td>
      <td colspan="3">TITAN XP</td>
      <td colspan="3">Tesla P100</td>
      <td colspan="3">RTX 2080Ti</td>
   </tr>
   <tr>
      <td>bs=1</td>
      <td>bs=4</td>
      <td>bs=8</td>
     <td>bs=1</td>
      <td>bs=4</td>
      <td>bs=8</td>
     <td>bs=1</td>
      <td>bs=4</td>
      <td>bs=8</td>
   </tr>
   <tr>
      <td>Jittor</td>
      <td>44</td>
      <td>54</td>
     <td>56</td>
     <td>42</td>
      <td>49</td>
     <td>52</td>
     <td>82</td>
      <td>98</td>
     <td>100</td>
   </tr>
   <tr>
      <td>Pytorch</td>
      <td>39</td>
      <td>48</td>
     <td>49</td>
     <td>35</td>
      <td>44</td>
     <td>44</td>
     <td>64</td>
      <td>71</td>
     <td>71</td>
   </tr>
  <tr>
      <td>Speedup</td>
      <td>1.13</td>
      <td>1.13</td>
     <td>1.14</td>
     <td>1.20</td>
      <td>1.11</td>
     <td>1.18</td>
     <td>1.28</td>
      <td>1.38</td>
     <td>1.41</td>
   </tr>
</table>

<table>
   <tr>
      <td rowspan="2"></td>
      <td colspan="3">Tesla V100 (16G PCI-E)</td>
      <td colspan="3">Tesla V100</td>
      <td colspan="3">RTX TITAN</td>
   </tr>
   <tr>
      <td>bs=1</td>
      <td>bs=4</td>
      <td>bs=8</td>
     <td>bs=1</td>
      <td>bs=4</td>
      <td>bs=8</td>
     <td>bs=1</td>
      <td>bs=4</td>
      <td>bs=8</td>
   </tr>
   <tr>
      <td>Jittor</td>
      <td>89</td>
      <td>115</td>
     <td>120</td>
     <td>88</td>
      <td>108</td>
     <td>113</td>
     <td>27</td>
      <td>74</td>
     <td>106</td>
   </tr>
   <tr>
      <td>Pytorch</td>
      <td>38</td>
      <td>75</td>
     <td>82</td>
     <td>10</td>
      <td>34</td>
     <td>53</td>
     <td>9</td>
      <td>15</td>
     <td>34</td>
   </tr>
   <tr>
      <td>Speedup</td>
      <td>2.34</td>
      <td>1.53</td>
     <td>1.46</td>
     <td>8.80</td>
      <td>3.18</td>
     <td>2.13</td>
     <td>3.00</td>
      <td>4.93</td>
     <td>3.12</td>
   </tr>
</table>

### Requirements
``` 
jittor
numpy
scipy
opencv-python
scikit-image
pytorch 1.0<=1.3
tqdm
yml

```

Pretrain model (based on ResNet50-FPN): <http://data.kaizhao.net/projects/deep-hough-transform/dht_r50_fpn_sel-c9a29d40.pth> (SEL dataset) and 
<http://data.kaizhao.net/projects/deep-hough-transform/dht_r50_nkl_d97b97138.pth> (NKL dataset / used in online demo)

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
cd deep-hough-transfrom
python data/prepare_data_JTLEE.py --root './data/ICCV2017_JTLEE_images/' --label './data/ICCV2017_JTLEE_gtlines_all' --save-dir './data/training/JTLEE_resize_100_100/' --list './data/training/JTLEE.lst' --prefix 'JTLEE_resize_100_100' --fixsize 400 --numangle 100 --numrho 100
```
For NKL dataset, you can download the dataset and put it to data dir. Then run python script to generate parametric space label.
```sh
cd deep-hough-transform
python data/prepare_data_NKL.py --root './data/NKL' --label './data/NKL' --save-dir './data/training/NKL_resize_100_100' --fixsize 400
```

### Forward
Generate visualization results and save coordinates to _.npy file.
```sh
CUDA_VISIBLE_DEVICES=0 python forward.py --model （your_best_model.pth） --tmp (your_result_save_dir)
```


### Citation
If our method/dataset are useful to your research, please consider to cite us:
```
@article{hu2020jittor,
  title={Jittor: a novel deep learning framework with meta-operators and unified graph execution},
  author={Hu, Shi-Min and Liang, Dun and Yang, Guo-Ye and Yang, Guo-Wei and Zhou, Wen-Yang},
  journal={Information Sciences},
  volume={63},
  number={222103},
  pages={1--21},
  year={2020}
}
```
```
@article{zhao2021deep,
  author    = {Kai Zhao and Qi Han and Chang-bin Zhang and Jun Xu and Ming-ming Cheng},
  title     = {Deep Hough Transform for Semantic Line Detection},
  journal   = {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
  year      = {2021},
  doi       = {10.1109/TPAMI.2021.3077129}
}
```
```
@inproceedings{eccv2020line,
  title={Deep Hough Transform for Semantic Line Detection},
  author={Qi Han and Kai Zhao and Jun Xu and Ming-Ming Cheng},
  booktitle={ECCV},
  pages={750--766},
  year={2020}
}
```

### License
This project is licensed under the [Creative Commons NonCommercial (CC BY-NC 3.0)](https://creativecommons.org/licenses/by-nc/3.0/) license where only
non-commercial usage is allowed. For commercial usage, please contact us.
