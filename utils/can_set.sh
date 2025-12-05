#!/bin/bash

sudo ip link set can0 down
sudo ip link set can0 type can bitrate 500000
sudo ip -d -s show can0
sudo ip link set can0 up