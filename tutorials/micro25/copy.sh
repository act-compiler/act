#!/bin/bash

set -e

cd "$(dirname "$0")"

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 {exercise1|exercise2|exercise3}" >&2
    exit 1
fi

ARG="$1"
NUM="${ARG%/}"
NUM="${NUM#exercise}"

case "$NUM" in
    1|2|3) EXERCISE="exercise$NUM" ;;
    *)
        echo "Invalid exercise: $ARG" >&2
        exit 1
    ;;
esac

cp -r exercise$NUM/boilerplate/* ../../
echo "Copied boilerplate files for exercise$NUM to docker working directory."
