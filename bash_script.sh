#!/bin/bash

# python main.py --lr=0.00003 --group_name="test"
# python main.py --lr=0.0001 --group_name="test"
# python main.py --lr=0.0003 --group_name="test"
# python main.py --lr=0.001 --group_name="test"
# python main.py --lr=0.003 --group_name="test"

python main.py --lr=0.0008 --group_name="long_run"
python main.py --lr=0.0003 --group_name="long_run"
python main.py --lr=0.0001 --group_name="long_run"
