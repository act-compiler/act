#!/bin/bash
set -e

# Default to UID/GID 1000 if not provided
USER_ID=${HOST_UID:-1000}
GROUP_ID=${HOST_GID:-1000}

# Start an interactive shell as root
bash -l

# After shell exit, fix permissions on workspace
if [ -d /workspace ]; then
    chown -R $USER_ID:$GROUP_ID /workspace
fi
