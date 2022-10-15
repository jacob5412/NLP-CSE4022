#!/bin/sh -e
set -x

isort --recursive  --force-single-line-imports --apply programs
autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place programs --exclude=__init__.py
black programs
isort --recursive --apply programs
