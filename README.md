# Optimizing Siamese-Based Networks through Multi-Level Mask-Guided Learning in VOT (MML Method)


![image](https://github.com/z55668910/MML/blob/main/figs/MML.PNG)
![image](https://github.com/z55668910/MML/blob/main/figs/head.png)



<div align="center">
  <img src="demo/output/12.gif" width="1280px" />
  <img src="demo/output/34.gif" width="1280px" />
  <p>Examples of SiamBAN outputs. The green boxes are the ground-truth bounding boxes of VOT2018, the yellow boxes are results yielded by SiamBAN.</p>
</div>




## Installation

Please find installation instructions in [`INSTALL.md`](INSTALL.md).

## Quick Start: Using MML

### Add MML to your PYTHONPATH

```bash
export PYTHONPATH=/path/to/MML:$PYTHONPATH
```

### Download models

Download models in [Model Zoo](MODEL_ZOO.md) and put the `model.pth` in the correct directory in experiments

### Webcam demo

```bash
python tools/demo.py \
    --config experiments/siamban_r50_l234/config.yaml \
    --snapshot experiments/siamban_r50_l234/model.pth
    # --video demo/bag.avi # (in case you don't have webcam)
```

### Download testing datasets

Download datasets and put them into `testing_dataset` directory. Jsons of commonly used datasets can be downloaded from [here](https://drive.google.com/drive/folders/10cfXjwQQBQeu48XMf2xc_W1LucpistPI) or [here](https://pan.baidu.com/s/1et_3n25ACXIkH063CCPOQQ), extraction code: `8fju`. If you want to test tracker on new dataset, please refer to [pysot-toolkit](https://github.com/StrangerZhang/pysot-toolkit) to setting `testing_dataset`. 

### Test tracker

```bash
cd experiments/siamban_r50_l234
python -u ../../tools/test.py 	\
	--snapshot model.pth 	\ # model path
	--dataset VOT2018 	\ # dataset name
	--config config.yaml	  # config file
```

The testing results will in the current directory(results/dataset/model_name/)

### Eval tracker

assume still in experiments/siamban_r50_l234

``` bash
python ../../tools/eval.py 	 \
	--tracker_path ./results \ # result path
	--dataset VOT2018        \ # dataset name
	--num 1 		 \ # number thread to eval
	--tracker_prefix 'model'   # tracker_name
```

###  Training :wrench:

See [TRAIN.md](TRAIN.md) for detailed instruction.

## Reference

- [Siamese Box Adaptive Network for Visual Tracking](https://arxiv.org/abs/2003.06761)
- [official repo](https://github.com/hqucv/siamban/tree/master)
