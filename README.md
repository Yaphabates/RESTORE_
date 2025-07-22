## Installation 
For installation and other package requirements, please follow the instructions detailed in [INSTALL.md](docs/INSTALL.md). 

## Data preparation
Please follow the instructions at [DATASETS.md](docs/DATASETS.md) to prepare all datasets.


#### (1) Base-to-Novel class generalization setting
The default training settings are provided in config file at `configs/trainers/MaPLe_FSA/vit_b16_c2_ep5_batch4_2ctx.yaml`. All hyper-parameters such as prompt length, prompt depth, etc., can be modified using this config file.

Below, we provide instructions to train MaPLe_FSA on imagenet. 


```bash
# Other possible dataset values includes [caltech101, food101, dtd, ucf101, oxford_flowers, oxford_pets, fgvc_aircraft, stanford_cars, sun397, eurosat]

# seed=1
# trains and evaluates on base classes
bash scripts/maple_fsa/base2new_train.sh imagenet 1 0.05 0.05 1
# evaluates on novel classes
bash scripts/maple_fsa/base2new_test.sh imagenet 1 0.05 0.05 1

```

#### Averaging results over 3 seeds: 
Once the above trainings and evaluations are completed, the `output/` directory should have the following structure:

```
output
|–– base2new/
|   |–– test_new/
|   |   |–– imagenet/
|   |   |   |–– shots_16/
|   |   |   |   |–– MaPLe_FSA/
|   |   |   |   |   |–– vit_b16_c2_ep5_batch4_2ctx/
|   |   |   |   |   |   |–– seed1/
|   |   |   |   |   |   |–– seed2/
|   |   |   |   |   |   |–– seed3/
|   |–– train_base/
|   |   |–– imagenet/
|   |   |   |–– shots_16/
|   |   |   |   |–– MaPLe_FSA/
|   |   |   |   |   |–– vit_b16_c2_ep5_batch4_2ctx/
|   |   |   |   |   |   |–– seed1/
|   |   |   |   |   |   |–– seed2/
|   |   |   |   |   |   |–– seed3/

```

The above steps can be repeated for other individual datasets.


#### (2) Cross-Dataset Transfer
We provide instructions to train our model on imageNet using all 1000 classes and then evaluating it directly on new downstream datasets.
We provide cross-dataset config for MaPLe_FSA: `configs/MaPLe_FSA/vit_b16_c2_ep5_batch4_2ctx_cross_datasets.yaml`.
* Firstly, train MaPLe on imagenet in few-shot manner (for all 3 seeds).

```bash
# seed=1 
bash scripts/maple_fsa/xd_train.sh imagenet 1 0.05 0.05 1
# seed=2 
bash scripts/maple_fsa/xd_train.sh imagenet 2 0.05 0.05 1
# seed=3 
bash scripts/maple_fsa/xd_train.sh imagenet 3 0.05 0.05 1
```

* Now evaluate ImageNet model on downstream datasets.

```bash
for SEED in 1 2 3
do
    bash scripts/maple_fsa/xd_test.sh caltech101 ${SEED} 0.05 0.05 1
    bash scripts/maple_fsa/xd_test.sh oxford_pets ${SEED} 0.05 0.05 1
    bash scripts/maple_fsa/xd_test.sh stanford_cars ${SEED} 0.05 0.05 1
done
```

#### (3) Domain Generalization 
We use imagenet to train our model for domain generalization experiments. The steps are similar to above cross-dataset experiments, however, model is evaluated on imagenet variants.
* Evaluate imageNet model on variants of imagenet (domain shift datasets).

```bash
for SEED in 1 2 3
do
    bash scripts/maple_fsa/xd_test.sh imagenetv2 ${SEED} 0.05 0.05 1
    bash scripts/maple_fsa/xd_test.sh imagenet_sketch ${SEED} 0.05 0.05 1
    bash scripts/maple_fsa/xd_test.sh imagenet_a ${SEED} 0.05 0.05 1
    bash scripts/maple_fsa/xd_test.sh imagenet_r ${SEED} 0.05 0.05 1
done
```