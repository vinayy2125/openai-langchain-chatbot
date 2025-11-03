
import os
import smtplib
from email.message import EmailMessage
from app.logger import get_logger
from typing import Optional
from dotenv import load_dotenv
from datetime import datetime
import requests
import markdown2

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
    """Try to claim the session email send using Redis SETNX.

    Returns True if this caller acquired the right to send the email (first caller),
    False if someone else already did.
    """
    try:
        if not redis_client:
            return True
        key = f"session:{session_id}:closure_email_sent"
        # set with nx=True and small TTL to avoid permanent key
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
    """Compose and send a closure email for a session with dynamic HTML UI."""
    try:
        cfg = _get_smtp_config()
        if not cfg.get("host") or not cfg.get("to_addr"):
            logger.info("[email_sender] SMTP or recipient not configured, skipping email send")
            return False
 
        proceed = _claim_email_sent(redis_client, session_id)
        if not proceed:
            logger.info(f"[email_sender] Closure email for session {session_id} already sent by another worker; skipping.")
            return False
 
        # --- Fetch Transcript + Session Meta ---
        transcript_lines = []
        user_region = "unknown"
        rows = []
        try:
            from app.api.v1.helpers import get_db_conn
            conn = get_db_conn()
            cur = conn.cursor()
 
            # Messages
            cur.execute("""
                SELECT role, content, created_at
                FROM messages
                WHERE session_id = %s
                ORDER BY created_at ASC
            """, (str(session_id),))
            rows = cur.fetchall()
            for r in rows:
                role, content, created_at = r
                ts = created_at.strftime("%Y-%m-%d %H:%M:%S") if hasattr(created_at, "strftime") else str(created_at)
                transcript_lines.append(f"[{ts}] {role.upper()}: {content}")
 
            # Session metadata
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
                    try:
                        resp = requests.get(f"https://ipapi.co/{ip}/json/", timeout=3)
                        if resp.status_code == 200:
                            data = resp.json()
                            city = data.get("city")
                            region = data.get("region") or data.get("region_name")
                            country = data.get("country_name")
                            user_region = ", ".join(filter(None, [city, region, country])) or "unknown"
                    except Exception as e:
                        logger.info(f"[email_sender] IP region lookup failed: {e}")
 
            cur.close()
            conn.close()
        except Exception as e:
            logger.warning(f"[email_sender] DB fetch failed for session {session_id}: {e}")
 
        # Fallback if no transcript
        if not transcript_lines and follow_up_manager:
            try:
                sh = follow_up_manager.get_conversation_history(session_id)
                for m in sh:
                    transcript_lines.append(f"[in-memory] {m.get('role', '').upper()}: {m.get('content', '')}")
            except Exception:
                pass
 
        # --- Duration Computation ---
        duration_str = "unknown"
        try:
            delta = None
            if session_created and session_last:
                delta = session_last - session_created
            elif rows:
                first, last = rows[0][2], rows[-1][2]
                delta = last - first
            if delta:
                total_seconds = int(delta.total_seconds())
                duration_str = str(timedelta(seconds=total_seconds))
        except Exception:
            pass
 
        # --- Markdown → HTML Transcript ---
        transcript_md = "\n".join(transcript_lines or ["(no transcript available)"])
        transcript_html = markdown2.markdown(
            transcript_md,
            extras=["fenced-code-blocks", "tables", "strike", "break-on-newline"]
        )
 
        # --- Dynamic Email HTML Template ---
        html = f"""
<html>
<body style='font-family:Segoe UI,Arial,sans-serif;background:#f7f9fb;padding:0;margin:0;'>
<div style='max-width:600px;margin:40px auto;background:#fff;border-radius:8px;
                      box-shadow:0 2px 8px #0001;padding:32px;'>
<h2 style='color:#2d6cdf;margin-top:0;'>Chat Session Summary</h2>
 
            <table style='width:100%;margin-bottom:24px;font-size:15px;'>
<tr><td style='color:#888;'>Session ID:</td><td>{session_id}</td></tr>
              {f"<tr><td style='color:#888;'>Reason:</td><td>{reason}</td></tr>" if reason else ""}
<tr><td style='color:#888;'>IP:</td><td>{ip or 'unknown'}</td></tr>
<tr><td style='color:#888;'>Region:</td><td>{user_region}</td></tr>
<tr><td style='color:#888;'>Browser:</td><td>{browser or 'unknown'}</td></tr>
<tr><td style='color:#888;'>Duration:</td><td>{duration_str}</td></tr>
</table>
 
            <div style='margin-bottom:12px;font-weight:bold;color:#2d6cdf;'>Chat Transcript:</div>
<div style='background:#f3f6fa;border-radius:6px;padding:16px 12px;
                        font-size:14px;line-height:1.7;color:#333;'>
              {transcript_html}
</div>
 
            <div style='margin-top:32px;color:#aaa;font-size:13px;text-align:center;'>
              This is an automated summary from <b>DITS Chatbot</b>.<br>
              © 2025 DITS Innovations. All rights reserved.
</div>
</div>
</body>
</html>
        """
 
        # --- Plaintext fallback ---
        text = (
            f"Session ID: {session_id}\n"
            + (f"Reason: {reason}\n" if reason else "")
            + f"IP: {ip or 'unknown'}\n"
            + f"Region: {user_region}\n"
            + f"Browser: {browser or 'unknown'}\n"
            + f"Duration: {duration_str}\n\nChat transcript:\n"
            + "\n".join(transcript_lines or ["(no transcript available)"])
        )
 
        # --- Build Email ---
        msg = EmailMessage()
        msg["Subject"] = f"Chat session summary: {session_id}"
        msg["From"] = cfg.get("from_addr") or cfg.get("user")
        msg["To"] = to_addr or cfg.get("to_addr")
        msg.set_content(text)
        msg.add_alternative(html, subtype="html")
 
        # --- Send via SMTP ---
        host, port = cfg.get("host"), int(cfg.get("port") or 25)
        user, password = cfg.get("user"), cfg.get("password")
        use_tls = cfg.get("use_tls", True)
 
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
 
        logger.info(f"[email_sender] Sent dynamic UI summary email for session {session_id} to {msg['To']}")
        return True
 
    except Exception as e:
        logger.exception(f"[email_sender] Failed to send session summary email for session {session_id}: {e}")
        return False