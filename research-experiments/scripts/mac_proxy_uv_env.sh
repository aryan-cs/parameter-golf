#!/usr/bin/env bash

_MAC_PROXY_UV_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_MAC_PROXY_UV_ENV_REPO_ROOT="$(cd "$_MAC_PROXY_UV_ENV_DIR/../.." && pwd)"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$_MAC_PROXY_UV_ENV_REPO_ROOT/research-experiments/cache/uv}"
mkdir -p "$UV_CACHE_DIR"
unset _MAC_PROXY_UV_ENV_DIR _MAC_PROXY_UV_ENV_REPO_ROOT
