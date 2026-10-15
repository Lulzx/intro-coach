#!/usr/bin/env bash
# Build the Core ML speech helper for a target triple and put it where Tauri's externalBin expects it.
#   scripts/build-native.sh [aarch64-apple-darwin|x86_64-apple-darwin]   (default: this Mac)
set -euo pipefail
cd "$(dirname "$0")/.."
triple="${1:-$(rustc --print host-tuple)}"
case "$triple" in
  aarch64-apple-darwin) arch=arm64 ;;
  x86_64-apple-darwin) arch=x86_64 ;;
  *) echo "Core ML helper is macOS only ($triple)"; exit 0 ;;
esac
args=(-c release --package-path native/coreml-asr --arch "$arch")
swift build "${args[@]}"
mkdir -p src-tauri/binaries
cp "$(swift build "${args[@]}" --show-bin-path)/coreml-asr" "src-tauri/binaries/coreml-asr-$triple"
echo "built src-tauri/binaries/coreml-asr-$triple"
