#!/bin/bash
python src/pretrain.py --learning_rate 0.001
python src/pretrain.py --learning_rate 0.0001
python src/pretrain.py --learning_rate 0.00001
python src/pretrain.py --learning_rate 0.000001
python src/pretrain.py --learning_rate 0.0000001
