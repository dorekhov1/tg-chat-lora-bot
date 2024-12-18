default:
    @just --list

prepare-data:
    python train/prepare_dataset.py

train:
    python train/train.py

up:
    docker compose up --build

down:
    docker compose down
