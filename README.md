# Improving Attention-based Few-shot Object Detection

### Introduction

This repository is the code implementation of Improving Attention-based Few-shot Object Detection. The source code the model architecture is contained within the `fsdet/modeling` directory while training code is contained within the `engine` and `tools` directories. This repository is based on [Frustratingly Simple Few-Shot Object Detection](https://github.com/ucbdrive/few-shot-object-detection) and the [Detectron2](https://github.com/facebookresearch/detectron2) framework. So while the model code is produced by me, the training and evaluation codes are taken from the parent repository.

For the theoretical part of my work, please read the pdf copy of the written report/paper "Improving Attention-based Few-shot Object Detection", which is contained in this directory. Please consider both as my "final artifact."

### Code structure

The main model architecture code can be found in `fsdet/modeling/meta_arch/rcnn.py`. The attention mechanism developed in my work is implemented in `fsdet/modeling/attention/cross_attention.py`. The code implementing the model detector head (i.e. the final layers of the detection network) can be found in `fsdet/modeling/roi_heads/roi_heads.py` and the Region Proposal Network code is located at `fsdet/modeling/proposal_generator/rpn.py`. It should be noted that only the code in `cross_attention.py` is written from scratch. The other codes are modified from the Faster RCNN implementation by Detectron2. 

The data processing code is located in `fsdet/data/meta_pascal_voc.py` and `fsdet/data/fs_data_mapper.py`. Note that I only implemented the data loading and generating code for the Pascal VOC dataset and not the LVIS or COCO dataset. `fs_data_mapper.py` is my data generator for training (in few-shot detection, training is done in episodes, where each episode requires k-shot instances of a certain category to be used as the support. This is what is implemented here). `meta_pascal_voc.py` is a modified data loader. 

### Installing prerequisites

Please create a python virtual environment and install the following

```
torch 1.6--1.8 (make sure to install cuda)
torchvision
```

Then, install detectron2 v0.4 [here](https://github.com/facebookresearch/detectron2/releases/tag/v0.4).

Finally, run 

```
python3 -m pip install -r requirements.txt
```

to install other dependencies. 

### Running the code

Training:

```
python3 -m tools.train_net --num-gpus #gpu \
        --config-file configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_base1.yaml
```

The training code should contain episodic evaluation of the model, every 10000 training steps. 

### Bibliography

```angular2html
[1] A. Ayub and A. R. Wagner. Tell me what this is: Few-shot incremental object learning by a robot. In 2020 IEEE/RSJ International Conference on Intelligent Robots and Sys- tems (IROS), pages 8344–8350, 2020. 1
[2] Z. Cai and N. Vasconcelos. Cascade R-CNN: high qual- ity object detection and instance segmentation. CoRR, abs/1906.09756, 2019. 2
[3] T.-I. Chen, Y.-C. Liu, H.-T. Su, Y.-C. Chang, Y.-H. Lin, J.-F. Yeh, W.-C. Chen, and W. H. Hsu. Dual-awareness attention for few-shot object detection. 2021. 1, 2, 3, 4
[4] J. Dai, Y. Li, K. He, and J. Sun. R-FCN: object detec- tion via region-based fully convolutional networks. CoRR, abs/1605.06409, 2016. 2
[5] J.Dai,H.Qi,Y.Xiong,Y.Li,G.Zhang,H.Hu,andY.Wei. Deformable convolutional networks, 2017. 2
[6] Q. Fan, W. Zhuo, C.-K. Tang, and Y.-W. Tai. Few-shot object detection with attention-rpn and multi-relation de- tector, 2019. 1, 2, 3
[7] Z. Fan, Y. Ma, Z. Li, and J. Sun. Generalized few-shot object detection without forgetting, 2021. 2
[8] C. Finn, P. Abbeel, and S. Levine. Model-agnostic meta- learning for fast adaptation of deep networks, 2017. 2
[9] Z. Li, C. Peng, G. Yu, X. Zhang, Y. Deng, and J. Sun.
Light-head r-cnn: In defense of two-stage object detector, 2017. 2
4[10] T.-Y. Lin, P. Dolla ́r, R. Girshick, K. He, B. Hariharan, and S. Belongie. Feature pyramid networks for object detec- tion, 2016. 4
[11] W. Liu, D. Anguelov, D. Erhan, C. Szegedy, S. E. Reed, C. Fu, and A. C. Berg. SSD: single shot multibox detector. CoRR, abs/1512.02325, 2015. 2
[12] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi. You only look once: Unified, real-time object detection, 2015. 2
[13] J. Redmon and A. Farhadi. YOLO9000: better, faster, stronger. CoRR, abs/1612.08242, 2016. 2
[14] J. Redmon and A. Farhadi. Yolov3: An incremental im- provement. CoRR, abs/1804.02767, 2018. 2
[15] A.A.Rusu,D.Rao,J.Sygnowski,O.Vinyals,R.Pascanu, S. Osindero, and R. Hadsell. Meta-learning with latent embedding optimization, 2018. 2
[16] L. B. Smith, S. S. Jones, B. Landau, L. Gershkoff-Stowe, and L. Samuelson. Object name learning provides on- the-job training for attention. Psychological Science, 13(1):13–19, 2002. 1
[17] J. Snell, K. Swersky, and R. S. Zemel. Prototypical net- works for few-shot learning, 2017. 2
[18] F. Sung, Y. Yang, L. Zhang, T. Xiang, P. H. S. Torr, and T. M. Hospedales. Learning to compare: Relation network for few-shot learning, 2017. 2
[19] A.Vaswani,N.Shazeer,N.Parmar,J.Uszkoreit,L.Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin. Attention is all you need, 2017. 4
[20] X. Wang, T. E. Huang, T. Darrell, J. E. Gonzalez, and F. Yu. Frustratingly simple few-shot object detection, 2020. 2
[21] Y.-X. Wang, D. Ramanan, and M. Hebert. Meta-learning to detect rare objects. In 2019 IEEE/CVF International Conference on Computer Vision (ICCV), pages 9924– 9933, 2019. 2
[22] X. Yan, Z. Chen, A. Xu, X. Wang, X. Liang, and L. Lin. Meta r-cnn : Towards general solver for instance-level few- shot learning, 2019. 2
[23] G. Zhang, Z. Luo, K. Cui, and S. Lu. Meta-detr: Image-level few-shot object detection with inter-class correlation exploitation, 2021. 2
```

