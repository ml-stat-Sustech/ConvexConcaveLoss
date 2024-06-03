
# Mitigating Privacy Risk in Membership Inference by Convex-Concave Loss

This repository is the offical implementation for the paper: [Mitigating Privacy Risk in Membership Inference by Convex-Concave Loss](https://arxiv.org/abs/2402.05453)

## Installation
```
cd ConvexConcaveLoss;
conda env create -f environment.yml;
conda activate ccl;
python setup.py install;
```

## Membership inference attacks
### Step 0: train target/shadow models
```
cd ConvexConcaveLoss/source/examples;
python train_models.py --mode target --training_type Normal --loss_type ccel --alpha 0.5 --beta 0.05 --gpu 0 --optimizer sgd --scheduler multi_step --epoch 300 --learning_rate 0.1;
python train_models.py --mode shadow --training_type Normal --loss_type ccel --alpha 0.5 --beta 0.05 --gpu 0 --optimizer sgd --scheduler multi_step --epoch 300 --learning_rate 0.1;
``` 
Note that you can also specify the `--loss_type` with different loss function, e.g., `ce`, `focal` and `ccql`.

### Step 1: perform membership inference attacks
```
python mia.py  --training_type Normal --loss_type ccel --attack_type metric-based --alpha 0.5 --beta 0.05 --gpu 0 --scheduler multi_step --epoch 300 --learning_rate 0.1;
```

## Citation
If you find this useful in your research, please consider citing:

```
@inproceedings{liu2024mitigating,
  title={Mitigating Privacy Risk in Membership Inference by Convex-Concave Loss},
  author={Zhenlong Liu and Lei Feng and Huiping Zhuang and Xiaofeng Cao and Hongxin Wei},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2024}
}
```

## Acknowledgements
Our implementation uses the source code from the following repositories:
[MLHospital](https://github.com/TrustAIResearch/MLHospita)