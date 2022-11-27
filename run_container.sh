#!/usr/bin/env bash

docker run \
    -p 8888:8888 \
    -v "$HOME/.config/gcloud/application_default_credentials.json":/gcp/creds.json:ro \
    --env GOOGLE_APPLICATION_CREDENTIALS=/gcp/creds.json \
    --gpus all \
    --mount type=bind,source="${LAYERJOT_HOME}",target=/layerjot \
    --mount type=bind,source="${LJ_DATA}",target=/data \
    --mount type=bind,source=/home,target=/home \
    --env="DISPLAY" \
    --rm --network host -it metaformer:latest
