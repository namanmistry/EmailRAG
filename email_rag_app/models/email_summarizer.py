import imaplib
import re
import string
import demoji
import sqlite3
from email.header import decode_header
from markdownify import markdownify as md
from urllib.parse import urlparse


def generate_response(prompt, image_path=None, model='meta/llama-4-maverick-17b-128e-instruct', api_key=None):
    import requests, base64, json

    invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json"
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 1.0,
        "top_p": 1.0,
        "stream": False
    }

    response = requests.post(invoke_url, headers=headers, json=payload)

    if response.status_code != 200:
        raise Exception(f"LLM request failed with status {response.status_code}: {response.text}")

    return response.json()

class EmailSummarizer:
    def __init__(self, email_id, password, imap_server='imap.gmail.com', db_path='emails.db'):
        self.email_id = email_id
        self.password = password
        self.imap_server = imap_server
        self.db_path = db_path
        self.mail = None
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self._ensure_summary_fields()
        self.connect()
        demoji.download_codes()

    def _ensure_summary_fields(self):
        self.cursor.execute("PRAGMA table_info(Emails)")
        existing_columns = {row[1] for row in self.cursor.fetchall()}

        if "summary" not in existing_columns:
            self.cursor.execute("ALTER TABLE Emails ADD COLUMN summary TEXT")
        
        if "category" not in existing_columns:
            self.cursor.execute("ALTER TABLE Emails ADD COLUMN category TEXT")

        self.conn.commit()


    def connect(self):
        self.mail = imaplib.IMAP4_SSL(self.imap_server)
        self.mail.login(self.email_id, self.password)

    def decode_header_value(self, val):
        decoded = decode_header(val)
        return ''.join(
            part.decode(enc or 'utf-8', errors='ignore') if isinstance(part, bytes) else part
            for part, enc in decoded
        )

    def clean_email_body(self, text):
        text = re.sub(r'^>.*$', '', text, flags=re.MULTILINE)
        text = re.split(r'(--\s|__+|\nOn\s.+?wrote:)', text)[0]
        url_pattern = re.compile(r'https?://\S+|www\.\S+')

        def replace_url(match):
            url = match.group(0)
            domain = urlparse(url).netloc or url
            return f"[Link: {domain}]"

        text = url_pattern.sub(replace_url, text)
        text = demoji.replace(text, '')
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        punctuation = string.punctuation.replace('[', '').replace(']', '')
        return text.translate(str.maketrans('', '', punctuation))

    def summarize_and_tag(self, body, nvidia_api_key=None):

        prompt = """
            You are an assistant that summarizes emails into a single sentence and classifies them into one of the following categories:
            ['urgent', 'important', 'promotion', 'personal', 'other'].

            EMAIL:
            {body}

            Return format:
            Summary: <short summary sentence>
            Category: <category>
            """
        prompt = prompt.format(body=body)
        result = generate_response(prompt=prompt, api_key=nvidia_api_key)
        result = result["choices"][0]["message"]["content"]
        summary_match = re.search(r"Summary:\s*(.*)", result)
        category_match = re.search(r"Category:\s*(.*)", result)
        return {
            "summary": summary_match.group(1).strip() if summary_match else "",
            "category": category_match.group(1).strip().lower() if category_match else "other"
        }

    def fetch_uncategorized_emails(self):
        self.cursor.execute("""
        SELECT email_id, body_plain FROM Emails
        WHERE summary IS NULL OR category IS NULL
        LIMIT 20
        """)
        return self.cursor.fetchall()

    def update_email_summary(self, email_id, summary, category):
        self.cursor.execute("""
        UPDATE Emails SET summary = ?, category = ? WHERE email_id = ?
        """, (summary, category, email_id))
        self.conn.commit()

    def process_uncategorized_emails(self, nvidia_api_key=None):
        rows = self.fetch_uncategorized_emails()
        results = []
        for email_id, body in rows:
            print(f"Processing email ID: {email_id}")
            cleaned = self.clean_email_body(body)
            result = self.summarize_and_tag(cleaned, nvidia_api_key)
            self.update_email_summary(email_id, result['summary'], result['category'])
            results.append({
                "email_id": email_id,
                "summary": result['summary'],
                "category": result['category']
            })
        return results


# Usage:
# summarizer = EmailSummarizer("your_email@gmail.com", "your_password")
# print(summarizer.process_uncategorized_emails())
