# Deployment Guide — Oracle Cloud Free Tier

The platform + your nodemailer service on a permanently-free server.

## Why Oracle Cloud Free Tier

| Option | Free? | Catch |
|---|---|---|
| **Oracle Cloud "Always Free"** ✅ | Yes, permanently | Signup needs a card (never charged); Mumbai region available |
| Render free tier | Yes | Service **sleeps after 15 min idle** — the 08:45 scheduler would miss; unusable for this |
| Railway | No | $5/month minimum |
| Fly.io | Mostly | Free allowances shrank; can exceed |
| AWS/GCP free tier | 12 months only | Then ₹3,000+/month |

Oracle's Always Free ARM tier (up to 4 OCPUs / 24GB RAM total) is the only
truly free option that runs 24/7 — which a scheduler needs.

---

## Step 1 — Create the server (~15 min, one time)

1. Sign up at **cloud.oracle.com/free** → choose home region **India West (Mumbai)**
   (lowest latency to NSE/Groww; cannot be changed later).
2. Console → Compute → Instances → **Create instance**:
   - Image: **Ubuntu 24.04**
   - Shape: **Ampere A1.Flex** (ARM) — 2 OCPUs, 12 GB RAM (within Always Free)
   - Add your SSH public key (`cat ~/.ssh/id_ed25519.pub`; generate with `ssh-keygen` if none)
3. Note the public IP. SSH in:
   ```bash
   ssh ubuntu@<PUBLIC_IP>
   ```
4. Open port 8000 (dashboard). In OCI Console: your instance's subnet →
   Security List → Add Ingress Rule: source `0.0.0.0/0`, TCP, dest port `8000`.
   Then on the VM:
   ```bash
   sudo iptables -I INPUT -p tcp --dport 8000 -j ACCEPT
   sudo netfilter-persistent save
   ```
   ⚠️ Do **not** open port 3001 — the mailer should stay localhost-only.

> If Ampere capacity is unavailable in Mumbai (common), retry at off-peak
> hours or use the smaller always-free AMD E2.1.Micro (1GB) — sufficient,
> since the LLM work happens on Groq's servers, not yours.

## Step 2 — Install the platform

```bash
# On the VM
sudo apt update && sudo apt install -y python3-venv python3-pip git nodejs npm

# Get the code (push your repo to GitHub first, private repo is fine)
git clone https://github.com/<you>/Billions.git
cd Billions
python3 -m venv .venv
.venv/bin/pip install -r requirements_platform.txt
```

## Step 3 — Secrets

Create `/home/ubuntu/Billions/.env` (never commit this):

```bash
nano .env
```
```
GROQ_API_KEY=gsk_...
GEMINI_API_KEY=...            # recommended second free provider
EMAIL_SERVICE_URL=http://localhost:3001/send
EMAIL_TO=piyushkhurana23@gmail.com
GROWW_API_KEY=...             # only needed when broker=groww
GROWW_API_SECRET=...
```
```bash
chmod 600 .env
```

## Step 4 — Deploy the nodemailer service

```bash
# Copy your mailer to the VM (from your Mac):
scp -r ~/path/to/your/mailer ubuntu@<PUBLIC_IP>:/home/ubuntu/mailer

# On the VM:
cd ~/mailer && npm install
# Put its SMTP credentials in its own .env / config as it expects.
# It must listen on 127.0.0.1:3001 (localhost only).
```

## Step 5 — Run both as system services (auto-start, auto-restart)

```bash
sudo cp ~/Billions/deploy/wealth-platform.service /etc/systemd/system/
sudo cp ~/Billions/deploy/mailer.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now mailer wealth-platform

# Check
systemctl status wealth-platform
curl localhost:8000/api/portfolio
```

The VM's clock is UTC but the scheduler is pinned to Asia/Kolkata in code —
cycles fire at 08:45 **IST** regardless of server timezone. Both services
restart automatically on crash and on VM reboot.

## Step 6 — Access the dashboard

- `http://<PUBLIC_IP>:8000` from any device.
- The dashboard has **no authentication** and the Run Cycle button is on it.
  For anything beyond a quick test — and definitely before live trading —
  don't leave it open to the internet. Two good options:
  - **Tailscale (recommended, free):** `curl -fsSL https://tailscale.com/install.sh | sh && sudo tailscale up`
    on the VM + the Tailscale app on your phone/laptop. Then close port 8000
    in the OCI security list and use `http://<tailscale-ip>:8000`. Private,
    encrypted, zero config.
  - Or SSH tunnel when needed: `ssh -L 8000:localhost:8000 ubuntu@<IP>`
    → open `http://localhost:8000` locally.

## Step 7 — Updating the code later

```bash
cd ~/Billions && git pull
.venv/bin/pip install -r requirements_platform.txt   # if deps changed
sudo systemctl restart wealth-platform
```

## Step 8 — Going live (when paper performance justifies it)

1. Add funds to your Groww account.
2. On the VM: edit `wealth_config.json` → `"broker": "groww"`, and set
   `"capital"` to the real amount.
3. `sudo systemctl restart wealth-platform`
4. Watch the log for `Groww connected`:
   `journalctl -u wealth-platform -f`
   If auth fails it **falls back to paper automatically** and says so.

### Capital reality check (do not skip)

| Capital | Verdict |
|---|---|
| ₹500–1,000 | **Non-functional.** 25%-per-stock cap (₹125–250) is below the ₹1,000 min-trade rule → the PM rejects every trade. Costs ~₹50/round-trip would be 5–10% drag anyway. |
| ₹5,000 | Bare minimum — rules barely clear; 1–2 positions max |
| ₹10,000–15,000 | Recommended starting point |

## Monitoring & maintenance

```bash
journalctl -u wealth-platform -f          # live platform logs
journalctl -u mailer -f                   # mailer logs
sqlite3 ~/Billions/data/platform.db 'select * from trades order by id desc limit 5;'
```

Backups: `data/` (SQLite DB, paper portfolio, agent memory) is everything.
```bash
# From your Mac, weekly:
scp -r ubuntu@<PUBLIC_IP>:~/Billions/data ./backup-$(date +%F)
```

## Total cost: ₹0/month
(Oracle Always Free VM + Groq/Gemini free LLM + Groww free API + SQLite + your nodemailer)



<!-- How to Deploy on Server - Both Applications -->
cd ~/Billions   # or wherever you cloned the repo

# Install any new Python deps (safe to run every deploy)
.venv/bin/pip install -r requirements_platform.txt

# Restart platform + mailer
sudo systemctl restart mailer
sudo systemctl restart wealth-platform

# Verify both are running
sudo systemctl status wealth-platform
sudo systemctl status mailer

journalctl -u wealth-platform -f
journalctl -u mailer -f