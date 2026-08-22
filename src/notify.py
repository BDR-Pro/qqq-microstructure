# Part of qqq-microstructure.
#
# One-way Telegram notifications through the user's bot (@ibkrclaudeaibot),
# stdlib only, best-effort by design: a dead network must never break the
# ritual, so send() catches everything and returns False. The token is never
# printed.
#
# Setup (once):
#   setx TELEGRAM_API  <bot token from @BotFather>
#   open @ibkrclaudeaibot in Telegram and send it /start (a bot cannot
#   message a person who has never messaged it)
#   run anything that notifies -- the first send discovers your chat id via
#   getUpdates and prints the  setx TELEGRAM_CHAT <id>  to pin it (getUpdates
#   only shows recent messages, so pinning beats re-discovering)

import json, os, urllib.parse, urllib.request


def _api(tok, method, **params):
    url = f'https://api.telegram.org/bot{tok}/{method}'
    data = urllib.parse.urlencode(params).encode()
    with urllib.request.urlopen(url, data, timeout=15) as r:
        return json.load(r)


def _chat(tok):
    cid = os.environ.get('TELEGRAM_CHAT')
    if cid:
        return cid
    for u in reversed(_api(tok, 'getUpdates').get('result', [])):
        c = (u.get('message') or u.get('channel_post') or {}) \
            .get('chat', {}).get('id')
        if c:
            print(f'telegram: chat {c} discovered -- run  '
                  f'setx TELEGRAM_CHAT {c}  to pin it')
            return str(c)
    print('telegram: no chat found -- open the bot and send /start, then '
          're-run')
    return None


def send(text):
    """Send text to the pinned/discovered chat. Never raises."""
    tok = os.environ.get('TELEGRAM_API')
    if not tok:
        return False
    try:
        cid = _chat(tok)
        if not cid:
            return False
        _api(tok, 'sendMessage', chat_id=cid, text=text[:4000])
        return True
    except Exception as ex:
        print(f'telegram: send failed ({type(ex).__name__})')
        return False
