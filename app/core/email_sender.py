import os
import smtplib
from email.message import EmailMessage
from app.logger import get_logger
from typing import Optional
from dotenv import load_dotenv
import requests
from pytz import timezone
 
load_dotenv()
logger = get_logger(__name__)
 
 
def _get_smtp_config():
    return {
        "host": os.getenv("SMTP_HOST"),
        "port": int(os.getenv("SMTP_PORT", "0")) if os.getenv("SMTP_PORT") else None,
        "user": os.getenv("SMTP_USER"),
        "password": os.getenv("SMTP_PASSWORD"),
        "use_tls": os.getenv("SMTP_USE_TLS", "true").lower() in ("1", "true", "yes"),
        "from_addr": os.getenv("EMAIL_FROM"),
        "to_addr": os.getenv("EMAIL_TO"),
    }
 
 
def _claim_email_sent(redis_client, session_id: str) -> bool:
    """Ensure only one worker sends the closure email."""
    try:
        if not redis_client:
            return True
        key = f"session:{session_id}:closure_email_sent"
        ok = redis_client.set(key, "1", nx=True, ex=60 * 60 * 24)
        return bool(ok)
    except Exception as e:
        logger.warning(f"[email_sender] Redis claim failed, allowing send to proceed: {e}")
        return True
 
 
def send_closure_email(
    session_id: str,
    follow_up_manager=None,
    reason: Optional[str] = None,
    redis_client=None,
    to_addr: Optional[str] = None,
    browser: Optional[str] = None,
    ip: Optional[str] = None,
    session_created=None,
    session_last=None,
) -> bool:
    """Compose and send a closure email for a chat session with modern UI."""
    try:
        cfg = _get_smtp_config()
        if not cfg.get("host") or not cfg.get("to_addr"):
            logger.info("[email_sender] SMTP or recipient not configured, skipping email send")
            return False
 
        # Avoid duplicate email sends
        if not _claim_email_sent(redis_client, session_id):
            logger.info(f"[email_sender] Closure email for session {session_id} already sent; skipping.")
            return False
 
        # --- Fetch transcript from DB ---
        transcript_blocks = []
        user_region = "unknown"
 
        try:
            from app.db.base import get_db_conn
            conn = get_db_conn()
            cur = conn.cursor()
 
            cur.execute("""
                SELECT role, content, created_at
                FROM messages
                WHERE session_id = %s
                ORDER BY created_at ASC
            """, (str(session_id),))
            rows = cur.fetchall()
            ist = timezone('Asia/Kolkata')
 
            # Loop through messages and format
            for r in rows:
                role, content, created_at = r
                ts = created_at.astimezone(ist).strftime("%Y-%m-%d %H:%M:%S IST") if hasattr(created_at, "astimezone") else str(created_at)
 
                block_class = "user" if role.lower() == "user" else "assistant"
                transcript_blocks.append(f"""
<div class="chat-block {block_class}">
<div class="chat-role">{role.capitalize()}:</div>
<div class="chat-message">{content}</div>
<div class="chat-time">{ts}</div>
</div>
                """)
 
            # --- Session metadata ---
            cur.execute("""
                SELECT browser, ip, created_at, last_interaction_at
                FROM sessions
                WHERE session_id = %s
                LIMIT 1
            """, (str(session_id),))
            srow = cur.fetchone()
            if srow:
                browser = browser or srow[0]
                ip = ip or srow[1]
                session_created = session_created or srow[2]
                session_last = session_last or srow[3]
 
                # Lookup region by IP
                if ip and ip not in ("unknown", None, ""):
                    import ipaddress
                    try:
                        ip_obj = ipaddress.ip_address(ip)
                        if not (ip_obj.is_private or ip_obj.is_loopback):
                            try:
                                resp = requests.get(f"https://ipapi.co/{ip}/json/", timeout=3)
                                if resp.status_code == 200:
                                    data = resp.json()
                                    city = data.get("city")
                                    region = data.get("region")
                                    country = data.get("country_name")
                                    user_region = ", ".join(filter(None, [city, region, country])) or "unknown"
                            except Exception:
                                pass
                        else:
                            user_region = "Private IP (not tracked)"
                    except Exception:
                        pass
 
            cur.close()
            conn.close()
 
        except Exception as e:
            logger.warning(f"[email_sender] DB fetch failed for session {session_id}: {e}")
 
        # --- Duration ---
        duration_str = "unknown"
        try:
            if session_created and session_last:
                delta = session_last - session_created
                total_seconds = int(delta.total_seconds())
                hours, rem = divmod(total_seconds, 3600)
                minutes, seconds = divmod(rem, 60)
                duration_str = f"{hours:02}:{minutes:02}:{seconds:02}"
        except Exception:
            pass
 
        # --- Build transcript HTML ---
        transcript_html = "\n".join(transcript_blocks or ["<div class='chat-block'><div class='chat-message'>(no transcript available)</div></div>"])
 
        # --- Build HTML email ---
        html = f"""
<html>
<head>
<meta charset="UTF-8">
<style>
body {{
    font-family: 'Segoe UI', Arial, sans-serif;
    background: #f5f7fb;
    margin: 0;
    padding: 0;
    color: #333;
}}
.wrapper {{
    max-width: 650px;
    margin: 40px auto;
    background: #fff;
    border-radius: 10px;
    box-shadow: 0 3px 10px rgba(0,0,0,0.08);
    overflow: hidden;
}}
.header {{
    background: #2d6cdf;
    color: #fff;
    padding: 20px 28px;
    font-size: 20px;
    font-weight: 600;
}}
.content {{
    padding: 28px;
}}
.info-table {{
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 28px;
    font-size: 15px;
}}
.info-table td {{
    padding: 6px 4px;
    vertical-align: top;
}}
.info-label {{
    color: #777;
    font-weight: 500;
    width: 150px;
}}
.info-value {{
    color: #111;
}}
.section-title {{
    font-size: 16px;
    font-weight: 600;
    color: #2d6cdf;
    margin-bottom: 12px;
    border-left: 4px solid #2d6cdf;
    padding-left: 8px;
}}
.chat-transcript {{
    background: #f8faff;
    border: 1px solid #e0e6ed;
    border-radius: 8px;
    padding: 18px 16px;
    max-height: 400px;
    overflow-y: auto;
    font-size: 15px;
    line-height: 1.6;
}}
.chat-block {{
    margin-bottom: 18px;
    padding-bottom: 14px;
    border-bottom: 1px solid #e0e6ed;
}}
.chat-block:last-child {{
    border-bottom: none;
}}
.chat-role {{
    font-weight: 600;
    color: #2d6cdf;
    margin-bottom: 6px;
}}
.chat-message {{
    color: #333;
    background: #fff;
    border: 1px solid #e4e8ef;
    border-radius: 8px;
    padding: 10px 14px;
    white-space: pre-line;
    margin-bottom: 6px;
}}
.chat-block.user .chat-message {{
    background: #e8f0fe;
    border-color: #d2e3fc;
}}
.chat-time {{
    font-size: 12px;
    color: #999;
}}
.footer {{
    text-align: center;
    font-size: 13px;
    color: #888;
    border-top: 1px solid #eee;
    padding: 18px 12px;
    background: #fafbfc;
}}
.footer b {{
    color: #2d6cdf;
}}
</style>
</head>
<body>
<div class="wrapper">
<div class="header">Chat Session Summary</div>
<div class="content">
<table class="info-table">
<tr><td class="info-label">Session ID:</td><td class="info-value">{session_id}</td></tr>
{f"<tr><td class='info-label'>Reason:</td><td class='info-value'>{reason}</td></tr>" if reason else ""}
<tr><td class="info-label">IP:</td><td class="info-value">{ip or 'Unknown'}</td></tr>
<tr><td class="info-label">Region:</td><td class="info-value">{user_region}</td></tr>
<tr><td class="info-label">Browser:</td><td class="info-value">{browser or 'Unknown'}</td></tr>
<tr><td class="info-label">Duration:</td><td class="info-value">{duration_str}</td></tr>
</table>
 
<div class="section-title">Chat Transcript</div>
<div class="chat-transcript">
{transcript_html}
</div>
</div>
<div class="footer">
This is an automated summary from <b>DITS Chatbot</b>.<br>
© 2025 DITS Innovations. All rights reserved.
</div>
</div>
</body>
</html>
"""
 
        # Plaintext fallback
        import re
        def strip_html_tags(text):
            return re.sub('<[^<]+?>', '', text)
        text = (
            f"Session ID: {session_id}\n"
            + (f"Reason: {reason}\n" if reason else "")
            + f"IP: {ip or 'unknown'}\nRegion: {user_region}\nBrowser: {browser or 'unknown'}\nDuration: {duration_str}\n\nChat transcript:\n"
            + "\n".join([strip_html_tags(line) for line in (transcript_blocks or ["(no transcript available)"])])
        )
        
          # Send email
        msg = EmailMessage()
        msg["Subject"] = f"Chat Session Summary: {session_id}"
        msg["From"] = cfg.get("from_addr") or cfg.get("user")
        msg["To"] = to_addr or cfg.get("to_addr")
        msg.set_content(text)
        msg.add_alternative(html, subtype="html")
 
        host, port = cfg.get("host"), int(cfg.get("port") or 25)
        use_tls = cfg.get("use_tls", True)
        user, password = cfg.get("user"), cfg.get("password")
 
        if use_tls and port == 465:
            smtp = smtplib.SMTP_SSL(host, port, timeout=10)
        else:
            smtp = smtplib.SMTP(host, port, timeout=10)
            if use_tls:
                smtp.starttls()
 
        if user and password:
            smtp.login(user, password)
        smtp.send_message(msg)
        smtp.quit()
 
        logger.info(f"[email_sender] Sent summary email for session {session_id} to {msg['To']}")
        return True
 
    except Exception as e:
        logger.exception(f"[email_sender] Failed to send session summary email for {session_id}: {e}")
        return False