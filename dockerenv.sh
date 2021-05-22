#!/bin/bash
make docker_image
docker run --rm -v "$(readlink -f  $(pwd)):/app/" -p 8000:8000 doodle_me $@