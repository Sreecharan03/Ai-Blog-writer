"""Quick SMTP test — tries primary (Gmail) then fallback (Brevo)."""
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)

MAIL_TO = "charanch53030@gmail.com"

relays = [
    {
        "label": "Gmail (primary)",
        "host": os.getenv("SMTP_HOST", ""),
        "port": int(os.getenv("SMTP_PORT", "587")),
        "user": os.getenv("SMTP_USER", ""),
        "password": os.getenv("SMTP_PASS", ""),
        "mail_from": os.getenv("MAIL_FROM", os.getenv("SMTP_USER", "")),
    },
    {
        "label": "Brevo (fallback)",
        "host": os.getenv("SMTP_FALLBACK_HOST", ""),
        "port": int(os.getenv("SMTP_FALLBACK_PORT", "587")),
        "user": os.getenv("SMTP_FALLBACK_USER", ""),
        "password": os.getenv("SMTP_FALLBACK_PASS", ""),
        "mail_from": os.getenv("SMTP_FALLBACK_FROM", os.getenv("SMTP_FALLBACK_USER", "")),
    },
]

for relay in relays:
    if not relay["host"] or not relay["user"]:
        print(f"\n[{relay['label']}] — skipped (not configured)")
        continue

    print(f"\n{'='*55}")
    print(f"Testing: {relay['label']}")
    print(f"  Host : {relay['host']}:{relay['port']}")
    print(f"  User : {relay['user']}")
    print(f"  From : {relay['mail_from']}")
    print(f"  To   : {MAIL_TO}")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Sighnal SMTP Test — {relay['label']}"
    msg["From"] = relay["mail_from"]
    msg["To"] = MAIL_TO
    msg.attach(MIMEText(
        f"<p>SMTP test from Sighnal via <b>{relay['label']}</b>. If you see this, that relay is working.</p>",
        "html"
    ))

    try:
        with smtplib.SMTP(relay["host"], relay["port"], timeout=15) as s:
            s.ehlo()
            s.starttls()
            s.login(relay["user"], relay["password"])
            s.sendmail(relay["mail_from"], [MAIL_TO], msg.as_string())
        print(f"  RESULT: SUCCESS")
    except Exception as e:
        print(f"  RESULT: FAILED — {type(e).__name__}: {e}")
