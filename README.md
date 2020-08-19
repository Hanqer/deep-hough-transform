Code accompanying the paper "Deep Hough Transform for Semantic Line Detection" (ECCV2020).

[arXiv2003.04676](https://arxiv.org/abs/2003.04676) | [Online Demo](http://mc.nankai.edu.cn/dht) | [Project page](http://mmcheng.net/dhtline) | [New dataset](http://kaizhao.net/sl5k)

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
### Forward
Generate visualization results and save coordinates to _.npy file.
```sh
CUDA_VISIBLE_DEVICES=0 python forward.py --tmp (dir to save results)
```

### Evaluate
Test the EA-score on SEL dataset. After forwarding the model and get the coordinates files. Run the following command to produce EA-score.
```sh
python test.py --pred result/debug/visualize_test/(change to your onw path which includes _.npy files) --gt gt_path/include_txt
```

### License
Our source code is free for non-commercial usage. For comercial usage. please contact us.

