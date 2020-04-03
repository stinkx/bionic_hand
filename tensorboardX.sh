#!/bin/bash

clear
cd

while true; do
	echo "TensorboardX starting..."
	tensorboard --logdir Documents/Git_repos/bionic_hand/Decoder/runs
	sleep 10
done

