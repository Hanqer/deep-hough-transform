Code for paper "Deep Hough Transform for Semantic Line Detection" (ECCV2020).
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
Pretrain model (based on ResNet50-FPN): https://drive.google.com/file/d/1a6Rbu1Bslyo9sjNlUUdi7NnSTdRIWwS5/view?usp=sharing
### Forward
Generate visualization results and save coordinates to _.npy file.
```sh
CUDA_VISIBLE_DEVICES=0 python forward.py --model （your_best_model.pth） --tmp (your_result_save_dir)
```
### Test
Test the EA-score on SEL dataset. After forwarding the model and get the coordinates files. Run the following command to produce EA-score.
```sh
python test.py --pred result/debug/visualize_test/(change to your onw path which includes _.npy files) --gt gt_path/include_txt
```
