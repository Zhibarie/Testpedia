#!/usr/bin/env bash
set -euo pipefail

TAG="${1:?tag is required}"
ZIP_NAME="${2:?zip name is required}"

# Ensure the target ref exists in local checkout before archiving.
git rev-parse --verify "$TAG" >/dev/null 2>&1

git archive --format=zip --output "$ZIP_NAME" "$TAG" -- \
  . \
  ':(exclude).github' \
  ':(exclude).github/**'
