import torch 
model = 'robnet_large_v1'
seed = 1111
data = '/JZ20200509'
model_param = dict(C=36,
                   num_classes=10,
                   layers=20,
                   steps=4,
                   multiplier=4,
                   stem_multiplier=3,
                   share=True,
                   AdPoolSize=1)

std = 0.8
dataSet = 'CIFAR10'
# randomly chosen the number of architectures
number = 500
