#!/bin/bash

echo "Adding cache busters to submodules..."

timestamp=$(date +%s)

echo "# Cache buster $timestamp" >> submodules/gsplat/setup.py
echo "# Cache buster $timestamp" >> submodules/simple-knn/setup.py
echo "# Cache buster $timestamp" >> submodules/diff-gaussian-rasterization/setup.py

echo "Cache busters added. Now run: ./using_docker.sh -nc"