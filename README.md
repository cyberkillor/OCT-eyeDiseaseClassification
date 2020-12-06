# DIP-OCT-Classification

> This project uses current models (resnet and vgg) to do OCT eye-disease classification
>
> Its results can be used as Baseline of OCT research

```shell
pip install -r requirement.txt
python main.py -h
python Eval.py -h
```

Type these two commands in your command line to see help menu for this project.

Also you can use sklearn.metric to do your own metric measurement.

To complete this project, you may need to add folder `./results/{model_name}/` and `./img/{model_name}/` to your local position.

The data set link is https://jbox.sjtu.edu.cn/l/XH2KA7

---

Sample image:
![image](https://github.com/cyberkillor/DIP-OCT-Classification/blob/main/img/Best-cm-img8.png)
