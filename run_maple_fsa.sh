# #### SEED 1

CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_train.sh fgvc_aircraft 1 0.1 0.1 1
CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_test.sh fgvc_aircraft 1 0.1 0.1 1

CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_train.sh caltech101 1 0.1 0.1 1
CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_test.sh caltech101 1 0.1 0.1 1

CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_train.sh stanford_cars 1 0.1 0.1 1
CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_test.sh stanford_cars 1 0.1 0.1 1

CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_train.sh oxford_pets 1 0.01 0.1 1
CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_test.sh oxford_pets 1 0.01 0.1 1

CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_train.sh oxford_flowers 1 0.1 0.1 1
CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_test.sh oxford_flowers 1 0.1 0.1 1

CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_train.sh food101 1 0.1 0.1 1
CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_test.sh food101 1 0.1 0.1 1

CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_train.sh sun397 1 0.05 0.05 1
CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_test.sh sun397 1 0.05 0.05 1

CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_train.sh ucf101 1 0.05 0.05 1
CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_test.sh ucf101 1 0.05 0.05 1

CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_train.sh dtd 1 0.15 0.1 0.5
CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_test.sh dtd 1 0.15 0.1 0.5

CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_train.sh eurosat 1 0.01 0.1 1
CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_test.sh eurosat 1 0.01 0.1 1

CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_train.sh imagenet 1 0.05 0.05 1
CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_test.sh imagenet 1 0.05 0.05 1

# CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/xd_train.sh imagenet 1 0.01 0.1 1
# CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/xd_test.sh imagenet 1 
# CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/xd_test.sh caltech101 1
# CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/xd_test.sh dtd 1
# CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/xd_test.sh eurosat 1
# CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/xd_test.sh fgvc_aircraft 1
# CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/xd_test.sh food101 1
# CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/xd_test.sh oxford_flowers 1
# CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/xd_test.sh oxford_pets 1
# CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/xd_test.sh stanford_cars 1
# CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/xd_test.sh sun397 1
# CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/xd_test.sh ucf101 1   
# CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/xd_test.sh imagenetv2 1  
# CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/xd_test.sh imagenet_r 1  
# CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/xd_test.sh imagenet_sketch 1   
# CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/xd_test.sh imagenet_r 1  


#### SEED 2

CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_train.sh fgvc_aircraft 2 0.1 0.1 1
CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_test.sh fgvc_aircraft 2 0.1 0.1 1

CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_train.sh caltech101 2 0.1 0.1 1
CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_test.sh caltech101 2 0.1 0.1 1

CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_train.sh stanford_cars 2 0.1 0.1 1
CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_test.sh stanford_cars 2 0.1 0.1 1

CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_train.sh oxford_pets 2 0.01 0.1 1
CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_test.sh oxford_pets 2 0.01 0.1 1

CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_train.sh oxford_flowers 2 0.1 0.1 1
CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_test.sh oxford_flowers 2 0.1 0.1 1

CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_train.sh food101 2 0.1 0.1 1
CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_test.sh food101 2 0.1 0.1 1

CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_train.sh sun397 2 0.05 0.05 1
CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_test.sh sun397 2 0.05 0.05 1

CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_train.sh ucf101 2 0.05 0.05 1
CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_test.sh ucf101 2 0.05 0.05 1

CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_train.sh dtd 2 0.15 0.1 0.5
CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_test.sh dtd 2 0.15 0.1 0.5

CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_train.sh eurosat 2 0.01 0.1 1
CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_test.sh eurosat 2 0.01 0.1 1

CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_train.sh imagenet 2 0.05 0.05 1
CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_test.sh imagenet 2 0.05 0.05 1

# CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/xd_train.sh imagenet 2 0.01 0.1 1
# CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/xd_test.sh imagenet 2
# CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/xd_test.sh caltech101 2
# CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/xd_test.sh dtd 2
# CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/xd_test.sh eurosat 2
# CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/xd_test.sh fgvc_aircraft 2
# CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/xd_test.sh food101 2
# CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/xd_test.sh oxford_flowers 2
# CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/xd_test.sh oxford_pets 2
# CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/xd_test.sh stanford_cars 2
# CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/xd_test.sh sun397 2
# CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/xd_test.sh ucf101 2  
# CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/xd_test.sh imagenetv2 2  
# CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/xd_test.sh imagenet_r 2 
# CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/xd_test.sh imagenet_sketch 2
# CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/xd_test.sh imagenet_r 2

#### SEED 3


CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_train.sh fgvc_aircraft 3 0.1 0.1 1
CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_test.sh fgvc_aircraft 3 0.1 0.1 1

CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_train.sh caltech101 3 0.1 0.1 1
CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_test.sh caltech101 3 0.1 0.1 1

CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_train.sh stanford_cars 3 0.1 0.1 1
CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_test.sh stanford_cars 3 0.1 0.1 1

CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_train.sh oxford_pets 3 0.01 0.1 1
CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_test.sh oxford_pets 3 0.01 0.1 1

CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_train.sh oxford_flowers 3 0.1 0.1 1
CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_test.sh oxford_flowers 3 0.1 0.1 1

CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_train.sh food101 3 0.1 0.1 1
CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_test.sh food101 3 0.1 0.1 1

CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_train.sh sun397 3 0.05 0.05 1
CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_test.sh sun397 3 0.05 0.05 1

CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_train.sh ucf101 3 0.05 0.05 1
CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_test.sh ucf101 3 0.05 0.05 1

CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_train.sh dtd 3 0.15 0.1 0.5
CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_test.sh dtd 3 0.15 0.1 0.5

CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_train.sh eurosat 3 0.01 0.1 1
CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_test.sh eurosat 3 0.01 0.1 1

CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_train.sh imagenet 3 0.05 0.05 1
CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/base2new_test.sh imagenet 3 0.05 0.05 1

# CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/xd_train.sh imagenet 3 0.01 0.1 1
# CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/xd_test.sh imagenet 3
# CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/xd_test.sh caltech101 3
# CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/xd_test.sh dtd 3
# CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/xd_test.sh eurosat 3
# CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/xd_test.sh fgvc_aircraft 3
# CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/xd_test.sh food101 3
# CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/xd_test.sh oxford_flowers 3
# CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/xd_test.sh oxford_pets 3
# CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/xd_test.sh stanford_cars 3
# CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/xd_test.sh sun397 3
# CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/xd_test.sh ucf101 3
# CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/xd_test.sh imagenetv2 3
# CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/xd_test.sh imagenet_r 3
# CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/xd_test.sh imagenet_sketch 3
# CUDA_VISIBLE_DEVICES=1 bash scripts/maple_fsa/xd_test.sh imagenet_r 3