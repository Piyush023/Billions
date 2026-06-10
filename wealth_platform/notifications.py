"""Notification layer: email-first (your nodemailer service), Telegram fallback.

Configure in .env:
  EMAIL_SERVICE_URL=http://localhost:3001/send     # your nodemailer endpoint
  EMAIL_TO=piyushkhurana23@gmail.com               # recipient
  EMAIL_SERVICE_API_KEY=...                        # optional; sent as Bearer token

Expected nodemailer endpoint contract (adjust PAYLOAD_STYLE if yours differs):
  POST {EMAIL_SERVICE_URL}
  JSON: {"to": "...", "subject": "...", "text": "...", "html": "..."}

If EMAIL_SERVICE_URL is unset, falls back to Telegram
(TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID); if neither is configured,
notifications are logged and dropped — the trading cycle never blocks on
notification failures.
"""

import logging
import os

import requests

logger = logging.getLogger("wealth_platform.notifications")


class Notifier:
    def __init__(self):
        self.email_url = os.getenv("EMAIL_SERVICE_URL")
        self.email_to = os.getenv("EMAIL_TO")
        self.email_api_key = os.getenv("EMAIL_SERVICE_API_KEY")
        self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")

    def send(self, subject: str, message: str) -> bool:
        """Send a notification. Returns True if any channel succeeded."""
        if self.email_url and self.email_to:
            if self._send_email(subject, message):
                return True
            logger.warning("Email send failed; trying Telegram fallback")
        if self.telegram_token and self.telegram_chat_id:
            return self._send_telegram(f"{subject}\n\n{message}")
        logger.info("No notification channel configured; dropping: %s", subject)
        return False

    def _send_email(self, subject: str, message: str) -> bool:
        headers = {}
        if self.email_api_key:
            headers["Authorization"] = f"Bearer {self.email_api_key}"
        try:
            resp = requests.post(
                self.email_url,
                json={
                    "to": self.email_to,
                    "subject": subject,
                    "text": message,
                    "html": "<pre style='font-family:monospace'>"
                    + message.replace("&", "&amp;").replace("<", "&lt;")
                    + "</pre>",
                },
                headers=headers,
                timeout=20,
            )
            resp.raise_for_status()
            logger.info("Email sent: %s", subject)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("Email send failed: %s", exc)
            return False

    def _send_telegram(self, message: str) -> bool:
        try:
            resp = requests.post(
                f"https://api.telegram.org/bot{self.telegram_token}/sendMessage",
                json={"chat_id": self.telegram_chat_id, "text": message[:4000]},
                timeout=15,
            )
            resp.raise_for_status()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("Telegram send failed: %s", exc)
            return False
