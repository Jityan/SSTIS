# Dataset Organization

The dataset folder structure for the paper: "[Text-to-image synthesis with self-supervised learning](https://doi.org/10.1016/j.patrec.2022.04.010)" Yong Xuan Tan, Chin Poo Lee, Mai Neo, Kian Ming Lim

## Oxford
```
oxford
│
└───jpg
│   │   image_00001.jpg
│   │   ...
│   
└───train
│   │   file021.txt
│   │   ...
│   
└───test
│   │   file021.txt
│   │   ...
│   
└───text_c10
    │
    └───class_00001
    │   │   image_06734.txt
    │   │   ...
    │   ...
```

## CUB
```
cub
│
└───CUB_200_2011
│   │   attributes
│   │   ...
│   
└───train
│   │   char-CNN-RNN-embeddings.pickle
│   │   ...
│   
└───test
│   │   char-CNN-RNN-embeddings.pickle
│   │   ...
│   
└───text_c10
    │
    └───001.Black_footed_Albatross
    │   │   Black_Footed_Albatross_0001_796111.txt
    │   │   ...
    │   ...
```