#!/bin/bash

echo "Installing python dependencies"
pip install -r requirements.txt

apt update
apt install -y default-jre
