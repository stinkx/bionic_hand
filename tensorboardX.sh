#!/bin/bash

clear
cd

while true; do
	echo "TensorboardX starting..."
	tensorboard --logdir Documents/MyoKI/Decoder/runs
	sleep 10
done

