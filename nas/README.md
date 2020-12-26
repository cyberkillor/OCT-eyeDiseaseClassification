# NAS

Use following commands to see how to manage inputs.

```python
python Search.py -h
python train.py -h
python Eval.py -h
```

Sample running commands.

```python
# samples for searching
python Search.py -sp ./results/m50_gn -sn 50
python Search.py -sp ./results/m100 -sn 100

# samples for training
python train.py -lp ./results/m50/ -m 0
python train.py -lp ./results/m50/ -m 1 --init

# samples for evaluating
python Eval.py -lp ./results/m100/ -m 0 
python Eval.py -lp ./results/m100/ -m 1 --init
```

