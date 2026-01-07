# Podman Container Setup

The bot runs as a rootless Podman container managed by systemd via Quadlet.

## Files

- `Containerfile` - Container image definition
- `strofkabot.container` - Quadlet unit file (installed to `~/.config/containers/systemd/`)
- `.containerignore` - Files excluded from the image

## Commands

### Service Management
```bash
systemctl --user status strofkabot     # Check status
systemctl --user start strofkabot      # Start
systemctl --user stop strofkabot       # Stop
systemctl --user restart strofkabot    # Restart
```

### Logs
```bash
journalctl --user -u strofkabot -f     # Follow systemd logs
podman logs strofkabot                 # Container logs
podman logs -f strofkabot              # Follow container logs
```

### Rebuild After Code Changes
```bash
systemctl --user stop strofkabot
podman build -t strofkabot .
systemctl --user start strofkabot
```

### Debug / Manual Run
```bash
# Run interactively (stop service first)
systemctl --user stop strofkabot
podman run -it --rm --env-file .env -v ./data:/app/data:Z strofkabot

# Run with debug logging
podman run -it --rm --env-file .env -v ./data:/app/data:Z strofkabot python strofkabot/llumi.py --log-level DEBUG
```

### Reinstall Quadlet (after editing strofkabot.container)
```bash
cp strofkabot.container ~/.config/containers/systemd/
systemctl --user daemon-reload
systemctl --user restart strofkabot
```

## Initial Setup (already done)

```bash
# Build image
podman build -t strofkabot .

# Create quadlet directory and install
mkdir -p ~/.config/containers/systemd
cp strofkabot.container ~/.config/containers/systemd/

# Enable user services to run without login
loginctl enable-linger $USER

# Reload and start
systemctl --user daemon-reload
systemctl --user start strofkabot
```

## Data Persistence

The `data/` directory is mounted as a volume, so the SQLite database persists across container rebuilds.
