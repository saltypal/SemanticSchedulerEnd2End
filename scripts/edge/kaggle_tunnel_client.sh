#!/usr/bin/env bash
# PC side of the edge-tunnel notebook (run in WSL). Usage: bash scripts/edge/kaggle_tunnel_client.sh <host>.trycloudflare.com
# Installs cloudflared into ~/.local/bin once (no sudo), then opens SSH to the Kaggle session through the tunnel.
# Extra args after the host go to ssh, e.g.  ... <host> -L 8888:localhost:8888   (port forward)
set -euo pipefail
host="${1:?usage: kaggle_tunnel_client.sh <host>.trycloudflare.com [ssh args...]}"; shift
bin="$HOME/.local/bin/cloudflared"
if [ ! -x "$bin" ]; then
  mkdir -p "$HOME/.local/bin"
  curl -fsSL -o "$bin" https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
  chmod +x "$bin"
fi
exec ssh -o "ProxyCommand=$bin access ssh --hostname %h" -o StrictHostKeyChecking=accept-new \
  -o ServerAliveInterval=30 "root@$host" "$@"
