#!/usr/bin/env bash
#-p 8888:8888

docker run \
    -v "$HOME/.config/gcloud/application_default_credentials.json":/gcp/creds.json:ro \
    --env GOOGLE_APPLICATION_CREDENTIALS=/gcp/creds.json \
    --gpus all \
    --mount type=bind,source="${LAYERJOT_HOME}",target=/layerjot \
    --mount type=bind,source=/data,target=/data \
    --mount type=bind,source=/home,target=/home \
    --env="DISPLAY" \
    --rm --network host -it --shm-size=2G metaformer.forest:latest
