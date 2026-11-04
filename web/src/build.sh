#!/bin/sh
# Builds ../lialign.wasm. Needs clang with the wasm32 target and wasm-ld (brew install lld).
cd "$(dirname "$0")"
PATH="/opt/homebrew/opt/lld/bin:$PATH" clang --target=wasm32 -O3 -nostdlib -fno-builtin-memset \
  -Wl,--no-entry -Wl,--export-dynamic -Wl,-z,stack-size=1048576 -Wl,--initial-memory=33554432 \
  -o ../lialign.wasm lialign.c
