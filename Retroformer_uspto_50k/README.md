
# Retroformer

The default hyperparameters and source code are available at https://github.com/yuewan2/Retroformer. Both a reaction-class-free mode and reaction-class-conditioned modes are trained, with the Boolean hyperparameter known_class controlling whether reaction-class information is provided (True or False). In addition, the knowledge-distilled variant of Retroformer is trained; its key hyperparameters are std_encoder_num_layers and std_decoder_num_layers, both are set as 4.

This work corresponds to the paper: [Retroformer: Pushing the Limits of Interpretable End-to-end Retrosynthesis Transformer](https://proceedings.mlr.press/v162/wan22a/wan22a.pdf).

## Dependency:
Follow the below steps for dependency installation:
```
conda create -y -n retroformer tqdm
conda activate retroformer
conda install pytorch=1.10.1 torchvision cudatoolkit=11.0 -c pytorch
conda install -y rdkit -c conda-forge
```
Simply run ```pip install -e .``` for other dependencies. 

The `rdchiral` package is taken from [here](https://github.com/connorcoley/rdchiral) (no need to install it).


## Directory overview:
The overview of the full directory looks like this:
```
Retroformer_uspto_50k/
├─checkpoint_typed
├─checkpoint_untyped
├─data
├─intermediate
├─log
├─result
├─retroformer
│  ├─models
│  │  └─__pycache__
│  ├─rdchiral
│  │  ├─test
│  │  └─__pycache__
│  ├─utils
│  │  └─__pycache__
│  └─__pycache__
```



## Data:

 The raw reaction dataset are presented into the data directory. One can also create your own reaction dataset.



## Train:

- a) The model is trained with reaction class constraints:
```
bash 01_train.sh
```

- b) The model is trained without reaction class constraints:
```
bash 01_train_no_class.sh
```


# translate by different reaction class strategies

```
02_translate_class_0_beam_30.sh
02_translate_class_1_beam_30.sh
02_translate_class_3_beam_10.sh
02_translate_class_5_beam_6.sh
02_translate_class_10_beam_3.sh

04_translate_kd_class_0_beam_30.sh
04_translate_kd_class_1_beam_30.sh
04_translate_kd_class_3_beam_10.sh
04_translate_kd_class_5_beam_6.sh
04_translate_kd_class_10_beam_3.sh
```

you can download the checkpoint file at https://a.b.com

The file path are Data_and_checkpoint_files/Retroformer_uspto_50k/checkpoint_typed and Data_and_checkpoint_files/Retroformer_uspto_50k/checkpoint_untyped, put the checkpoint file in the same file name.


## Reference:

If you find our work useful, please cite by:
```
@InProceedings{pmlr-v162-wan22a,
  title = {Retroformer: Pushing the Limits of End-to-end Retrosynthesis Transformer},
  author = {Wan, Yue and Hsieh, Chang-Yu and Liao, Ben and Zhang, Shengyu},
  booktitle = {Proceedings of the 39th International Conference on Machine Learning},
  pages = {22475--22490},
  year = {2022},
  volume = {162},
  series = {Proceedings of Machine Learning Research},
  month = {17--23 Jul},
  publisher = {PMLR},
  pdf = {https://proceedings.mlr.press/v162/wan22a/wan22a.pdf},
  url = {https://proceedings.mlr.press/v162/wan22a.html}
}
```



