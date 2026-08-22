#!/usr/bin/env bash
# Part of qqq-microstructure.
#
# Headless IBKR on a Linux VM -- the light alternative to the Windows TWS GUI:
# IB Gateway (IBKR's minimal client, bundles its own JRE) kept logged in and
# alive by IBC (github.com/IbcAlpha/IBC) under a virtual display (Xvfb),
# wrapped in a systemd service that restarts it forever. PAPER ONLY by
# design: the trading mode is hard-set to paper here, matching orders.py's
# refusal of live accounts until the forward verdict (RESULTS 19/20).
#
# Security, before anything else:
#   - The IB API socket (4002) is UNAUTHENTICATED. It binds localhost by
#     default; never port-forward it, never open it in the VM firewall or
#     cloud security group. Everything that needs it (orders.py, fills.py)
#     runs ON this VM.
#   - Credentials go into ~/ibc/config.ini chmod 600, prompted below, never
#     into this script or the repo. Use the PAPER username (IBKR paper
#     logins normally skip two-factor; live cannot run headless anyway).
#
# Tested shape: Ubuntu/Debian x64. Run as the user that will run the cron
# jobs:   bash install_ibkr.sh
#
# After it finishes: log out of paper on any other machine first (one paper
# session at a time), then  sudo systemctl start ibgateway  and run the
# smoke test it prints. The daily crontab lives in .github/workflows/roi.yml.

set -euo pipefail

IBC_VER="${IBC_VER:-3.20.0}"
GW_URL="https://download2.interactivebrokers.com/installers/ibgateway/stable-standalone/ibgateway-stable-standalone-linux-x64.sh"
IBC_URL="https://github.com/IbcAlpha/IBC/releases/download/${IBC_VER}/IBCLinux-${IBC_VER}.zip"
SUDO=""; [ "$(id -u)" != 0 ] && SUDO="sudo"

echo "== 1/6 system packages (xvfb, unzip, curl, python3-pip) =="
$SUDO apt-get update -qq
$SUDO apt-get install -y -qq xvfb unzip curl python3-pip libxtst6 libxrender1 libxi6

echo "== 2/6 IB Gateway (stable, standalone) =="
if [ ! -d "$HOME/Jts/ibgateway" ]; then
    curl -fL --retry 3 -o /tmp/ibgw.sh "$GW_URL"
    sh /tmp/ibgw.sh -q -dir "$HOME/Jts/ibgateway/tmp" || sh /tmp/ibgw.sh -q
fi
GW_VER=$(ls "$HOME/Jts/ibgateway" | grep -E '^[0-9]+' | sort -V | tail -1)
[ -n "$GW_VER" ] || { echo "IB Gateway install not found under ~/Jts/ibgateway"; exit 1; }
echo "   gateway version: $GW_VER"

echo "== 3/6 IBC $IBC_VER =="
mkdir -p "$HOME/ibc"
curl -fL --retry 3 -o /tmp/ibc.zip "$IBC_URL"
unzip -oq /tmp/ibc.zip -d "$HOME/ibc"
chmod +x "$HOME"/ibc/*.sh "$HOME"/ibc/scripts/*.sh

echo "== 4/6 credentials (paper) -> ~/ibc/config.ini (chmod 600) =="
if [ ! -f "$HOME/ibc/config.ini" ] || ! grep -q '^IbLoginId=.' "$HOME/ibc/config.ini"; then
    read -rp "IBKR PAPER username: " IB_USER
    read -rsp "IBKR PAPER password: " IB_PASS; echo
    cat > "$HOME/ibc/config.ini" <<CFG
IbLoginId=${IB_USER}
IbPassword=${IB_PASS}
TradingMode=paper
AcceptNonBrokerageAccountWarning=yes
ReadOnlyApi=no
OverrideTwsApiPort=4002
AcceptIncomingConnectionAction=accept
CFG
    chmod 600 "$HOME/ibc/config.ini"
else
    echo "   keeping existing ~/ibc/config.ini"
fi

echo "== 5/6 systemd service =="
cat > /tmp/ibgateway.service <<UNIT
[Unit]
Description=IB Gateway (paper) via IBC under Xvfb
After=network-online.target

[Service]
Type=simple
User=$(id -un)
ExecStart=/usr/bin/xvfb-run -a $HOME/ibc/scripts/ibcstart.sh ${GW_VER} --gateway --mode=paper --ibc-path=$HOME/ibc --ibc-ini=$HOME/ibc/config.ini --tws-path=$HOME/Jts
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
UNIT
$SUDO mv /tmp/ibgateway.service /etc/systemd/system/ibgateway.service
$SUDO systemctl daemon-reload
$SUDO systemctl enable ibgateway

echo "== 6/6 python side =="
pip3 install --quiet --user ib_insync numpy pandas yfinance
grep -q 'IBKR_API=' "$HOME/.profile" 2>/dev/null || \
    echo 'export IBKR_API=127.0.0.1:4002:7' >> "$HOME/.profile"

cat <<'DONE'

done. Next, in order:
  1. log the paper account OUT everywhere else (one paper session at a time),
  2.   sudo systemctl start ibgateway     (first login takes ~1-2 min)
  3. smoke test:
       python3 - <<'EOF'
from ib_insync import IB
ib = IB(); ib.connect('127.0.0.1', 4002, clientId=99)
print('accounts:', ib.managedAccounts()); ib.disconnect()
EOF
     (must print a D-prefixed account)
  4. clone the repo, set TELEGRAM_API / TELEGRAM_CHAT in this user's
     environment, and install the crontab from .github/workflows/roi.yml.

If step 3 hangs: journalctl -u ibgateway -e  usually shows a login dialog
IBC is answering; a version bump of IB Gateway changes GW_VER -- re-run this
script, it is idempotent.
DONE
