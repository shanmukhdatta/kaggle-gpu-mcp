"""Email notification via STARTTLS or SMTP_SSL; terminal jobs only."""
import smtplib
import ssl
from email.message import EmailMessage

from .config import Settings


class Mailer:
    def __init__(self, settings: Settings):
        self.s = settings

    def enabled(self) -> bool:
        return bool(self.s.email_to and self.s.smtp_host and self.s.smtp_sender)

    def send(self, subject: str, message: str) -> bool:
        if not self.enabled():
            return False
        mail = EmailMessage()
        mail["From"] = self.s.smtp_sender
        mail["To"] = self.s.email_to
        mail["Subject"] = subject[:150].replace("\n", " ").replace("\r", " ")
        mail.set_content(message[:8000])
        context = ssl.create_default_context()
        smtp_cls = smtplib.SMTP_SSL if self.s.smtp_ssl else smtplib.SMTP
        kwargs = {"context": context} if self.s.smtp_ssl else {}
        with smtp_cls(self.s.smtp_host, self.s.smtp_port, timeout=30, **kwargs) as smtp:
            if not self.s.smtp_ssl:
                smtp.ehlo()
                smtp.starttls(context=context)
                smtp.ehlo()
            if self.s.smtp_user:
                smtp.login(self.s.smtp_user, self.s.smtp_password)
            smtp.send_message(mail)
        return True
