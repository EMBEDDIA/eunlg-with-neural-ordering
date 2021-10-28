#!/bin/bash

echo "Checking for data"
wget -nc -nv -O splits/cphi.cache http://perfectlylegit.website/eunlg-data/cphi.cache
wget -nc -nv -O splits/env.cache http://perfectlylegit.website/eunlg-data/env.cache
wget -nc -nv -O splits/health_cost.cache http://perfectlylegit.website/eunlg-data/health_cost.cache
wget -nc -nv -O splits/health_funding.cache http://perfectlylegit.website/eunlg-data/health_funding.cache
exec "$@"
