#!/bin/bash
set -e

echo "Stopping strofkabot..."
systemctl --user stop strofkabot

echo "Building container..."
podman build -t strofkabot .

echo "Cleaning up dangling images..."
podman image prune -f

echo "Starting strofkabot..."
systemctl --user start strofkabot

echo "Deploy complete. Checking status..."
systemctl --user status strofkabot --no-pager
