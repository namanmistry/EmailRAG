import imaplib
import email
import os
import sqlite3
import demoji
import re
import string
from datetime import datetime, timezone
from urllib.parse import urlparse
from email.header import decode_header
from email.utils import parsedate_to_datetime, parseaddr
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

class Indexing:
    def __init__(self, imap_server='imap.gmail.com', db_path='emails.db'):
        self.imap_server = imap_server
        self.db_path = db_path
        self.mail = None

        # Setup DB connection and tables
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self._create_tables()

    def _create_tables(self):
        self.cursor.executescript('''
        CREATE TABLE IF NOT EXISTS Emails (
            email_id INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id TEXT UNIQUE,
            in_reply_to TEXT,
            references_list TEXT,
            subject TEXT,
            body_plain TEXT,
            sent_date TEXT,
            sent_date_utc TEXT,
            received_date TEXT,
            received_date_utc TEXT,
            from_name TEXT,
            from_email_id TEXT,
            to_name TEXT,
            to_email_id TEXT,
            cc_addresses TEXT,
            bcc_addresses TEXT,
            folder TEXT,
            flags TEXT,
            has_attachments INTEGER
        );

        CREATE TABLE IF NOT EXISTS Threads (
            thread_id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject TEXT,
            created_date TEXT,
            updated_date TEXT
        );

        CREATE TABLE IF NOT EXISTS EmailThreads (
            email_id INTEGER,
            thread_id INTEGER,
            FOREIGN KEY(email_id) REFERENCES Emails(email_id),
            FOREIGN KEY(thread_id) REFERENCES Threads(thread_id)
        );

        CREATE TABLE IF NOT EXISTS Attachments (
            attachment_id INTEGER PRIMARY KEY AUTOINCREMENT,
            email_id INTEGER,
            file_name TEXT,
            mime_type TEXT,
            file_size INTEGER,
            storage_path TEXT,
            FOREIGN KEY(email_id) REFERENCES Emails(email_id)
        );

        CREATE TABLE IF NOT EXISTS Participants (
            participant_id INTEGER PRIMARY KEY AUTOINCREMENT,
            email_address TEXT UNIQUE,
            name TEXT
        );

        CREATE TABLE IF NOT EXISTS EmailParticipants (
            email_id INTEGER,
            participant_id INTEGER,
            role TEXT,
            FOREIGN KEY(email_id) REFERENCES Emails(email_id),
            FOREIGN KEY(participant_id) REFERENCES Participants(participant_id)
        );
        ''')
        self.conn.commit()

    def login_email(self, email, password):
        self.mail = imaplib.IMAP4_SSL(self.imap_server)
        self.mail.login(email, password)

    def fetch_emails(self, folder='INBOX', max_emails=20):
        demoji.download_codes()
        self.mail.select(folder)
        status, messages = self.mail.search(None, 'ALL')
        email_ids = messages[0].split()

        for num in email_ids[:max_emails]:
            status, data = self.mail.fetch(num, '(RFC822)')
            if status != 'OK':
                continue

            msg = email.message_from_bytes(data[0][1])
            self._store_email_in_db(msg)

        self.conn.commit()
        self.cursor.execute("SELECT * FROM Emails")
        return [email.message_from_string(row[4]) for row in self.cursor.fetchall()]  # Return subject as dummy

    def _store_email_in_db(self, msg):
        message_id = msg.get('Message-ID')
        in_reply_to = msg.get('In-Reply-To')
        references = msg.get('References')
        subject = self.remove_emojis(self.decode_mime_header(msg.get('Subject', '')))
        subject = self.selective_lowercase(subject)
        from_address = self.decode_mime_header(msg.get('From', ''))
        from_name, from_email_id = parseaddr(from_address)
        to_addresses = self.decode_mime_header(msg.get('To', ''))
        to_name, to_email_id = parseaddr(to_addresses)
        cc_addresses = self.decode_mime_header(msg.get('Cc', ''))
        bcc_addresses = self.decode_mime_header(msg.get('Bcc', ''))
        sent_date = msg.get('Date')
        sent_date_utc, sent_date_local = self.store_email_date(sent_date) if sent_date else (None, None)
        received_date = msg.get('Date')
        received_date_utc, received_date_local = self.store_email_date(received_date) if received_date else (None, None)
        flags = ''
        folder = 'INBOX'
        has_attachments = 0

        body_plain = ''
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get('Content-Disposition'))

            if content_type == 'text/plain' and 'attachment' not in content_disposition:
                body_plain += part.get_payload(decode=True).decode(errors='ignore')
            elif 'attachment' in content_disposition:
                has_attachments = 1

        self.cursor.execute('''
            INSERT OR IGNORE INTO Emails (
                message_id, in_reply_to, references_list, subject, body_plain,
                sent_date, sent_date_utc, received_date, received_date_utc,
                from_name, from_email_id, to_name, to_email_id, cc_addresses,
                bcc_addresses, folder, flags, has_attachments
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            message_id, in_reply_to, references, subject, self.clean_plain_text_email(body_plain),
            sent_date_local, sent_date_utc, received_date_local, received_date_utc,
            from_name, from_email_id, to_name, to_email_id, cc_addresses,
            bcc_addresses, folder, flags, has_attachments
        ))
        self.conn.commit()

    def decode_mime_header(self, header_value):
        decoded_fragments = decode_header(header_value)
        return ''.join(
            fragment.decode(encoding or 'utf-8', errors='replace') if isinstance(fragment, bytes) else fragment
            for fragment, encoding in decoded_fragments
        )

    def clean_plain_text_email(self, text):
        text = re.sub(r'^>.*$', '', text, flags=re.MULTILINE)
        text = re.split(r'(--\s|__+|\nOn\s.+?wrote:)', text)[0]
        url_pattern = re.compile(r'https?://\S+|www\.\S+')

        def replace_url(match):
            url = match.group(0)
            domain = urlparse(url).netloc or url
            return f"[Link: {domain}]"

        text = url_pattern.sub(replace_url, text)
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        punct = string.punctuation.replace('[', '').replace(']', '')
        text = text.translate(str.maketrans('', '', punct))
        return text

    def remove_emojis(self, text):
        return demoji.replace(text, "")

    def selective_lowercase(self, text):
        def is_acronym(token):
            return re.findall(r'\b(?:[A-Z]\\.?){2,}\b', token)

        tokens = text.split()
        return ' '.join([token if is_acronym(token) else token.lower() for token in tokens])

    def store_email_date(self, date_str, user_timezone_str='America/Phoenix'):
        try:
            dt = parsedate_to_datetime(date_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            utc_dt = dt.astimezone(timezone.utc)
            user_tz = timezone.utc  # Replace if you want to localize
            local_dt = utc_dt.astimezone(user_tz)
            return utc_dt.isoformat(), local_dt.isoformat()
        except:
            return None, None

    def extract_datetime_components(self, date_str):
        try:
            dt = datetime.fromisoformat(date_str)
            return {
                "date": dt.date().isoformat(),
                "time": dt.time().isoformat(),
                "year": dt.year,
                "month": dt.month,
                "day": dt.day,
                "weekday": dt.strftime("%A"),
                "hour": dt.hour,
                "minute": dt.minute,
                "second": dt.second,
                "utc_offset": dt.strftime("%z"),
                "is_morning": dt.hour < 12,
                "time_bucket": (
                    "early_morning" if dt.hour < 6 else
                    "morning" if dt.hour < 12 else
                    "afternoon" if dt.hour < 17 else
                    "evening" if dt.hour < 21 else
                    "night"
                )
            }
        except:
            return {}

    def chunk_email(self, emails):
        documents = []
        for msg in emails:
            subject = decode_header(msg.get('Subject', ''))[0][0]
            subject = subject.decode() if isinstance(subject, bytes) else subject
            body_plain = self.clean_plain_text_email(msg.as_string())
            from_name, from_email = parseaddr(msg.get('From', ''))
            to_name, to_email = parseaddr(msg.get('To', ''))
            sent_date = msg.get('Date')
            sent_dt = parsedate_to_datetime(sent_date).astimezone(timezone.utc).isoformat() if sent_date else ''
            metadata = {
                "subject": subject,
                "from_name": from_name,
                "from_email_id": from_email,
                "to_name": to_name,
                "to_email_id": to_email,
                "sent_date": sent_dt,
                **self.extract_datetime_components(sent_dt)
            }
            documents.append(Document(page_content=body_plain, metadata=metadata))

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True
        )

        chunked_docs = splitter.split_documents(documents)
        for i, doc in enumerate(chunked_docs):
            doc.metadata["chunk_index"] = i

        return chunked_docs

    def create_vector_store(self, chunked_documents, embedding_model, persist_dir="./email_chroma_db"):
        if os.path.exists(persist_dir) and os.listdir(persist_dir):
            vector_store = Chroma(
                persist_directory=persist_dir,
                embedding_function=embedding_model
            )
        else:
            vector_store = Chroma.from_documents(
                documents=chunked_documents,
                embedding=embedding_model,
                persist_directory=persist_dir
            )
            vector_store.persist()
        return vector_store