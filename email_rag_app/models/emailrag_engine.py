import sqlite3



import sqlite3
import os
from datetime import datetime
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

def extract_datetime_components(date_str):
    """Extract datetime parts for metadata enrichment."""
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
    except Exception:
        return {}

def load_emails_from_db(db_path='C:\Personal\RAGEmail\emails.db'):
    """Load full emails from SQLite without chunking."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT subject, body_plain, from_name, from_email_id, 
               to_name, to_email_id, sent_date, folder, has_attachments, message_id 
        FROM Emails
    """)
    rows = cursor.fetchall()
    conn.close()
    return rows

def process_emails_as_whole(rows):
    """Process emails without chunking - one document per email."""
    documents = []
    for row in rows:
        subject, body_plain, from_name, from_email_id, to_name, to_email_id, sent_date, folder, has_attachments, message_id = row
        time_parts = extract_datetime_components(sent_date)
        metadata = {
            "subject": subject,
            "from_name": from_name,
            "from_email_id": from_email_id,
            "to_name": to_name,
            "to_email_id": to_email_id,
            "sent_date": sent_date,
            "folder": folder,
            "has_attachments": bool(has_attachments),
            "message_id": message_id,
            **time_parts
        }
        content = body_plain if body_plain else ""
        documents.append(Document(page_content=content, metadata=metadata))
    return documents

from tqdm import tqdm

def create_email_vectorstore(embedding_model, persist_dir=r"C:\Personal\RAGEmail\email_chroma_db_new_whole_email"):
    """Create and persist vector store for emails"""
    
    # Check if the database already exists
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        print("Loading existing vector database...")
        vector_store = Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding_model
        )
    else:
        print("error")
    
    return vector_store
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2")

# Create or load vector store
vector_store = create_email_vectorstore(embedding_model)
table_schema = """You are working with a relational database table named "Emails" that stores parsed email metadata and content. Below is the schema description:

Table: Emails

- email_id (INTEGER, Primary Key, Auto-Increment): Unique internal identifier for each email record.
- message_id (TEXT, Unique): Globally unique identifier of the email message.
- in_reply_to (TEXT): The message ID this email is replying to, if applicable.
- references_list (TEXT): List of message IDs referenced in the email thread.
- subject (TEXT): The subject line of the email.
- body_plain (TEXT): The plain text body content of the email, without HTML formatting.
- sent_date (TEXT): The date and time when the email was sent, in the format "YYYY-MM-DD HH:MM:SS±HH:MM" (e.g., "2022-05-03 05:39:01-07:00").
- sent_date_utc (TEXT): The UTC equivalent of the sent date.
- received_date (TEXT): The date and time when the email was received, in the same timestamp format.
- received_date_utc (TEXT): The UTC equivalent of the received date.
- from_name (TEXT): The display name of the sender.
- from_email_id (TEXT): The email address of the sender.
- to_name (TEXT): The display name of the primary recipient(s).
- to_email_id (TEXT): The email address(es) of the primary recipient(s).
- cc_addresses (TEXT): Comma-separated list of CC (carbon copy) recipient email addresses.
- bcc_addresses (TEXT): Comma-separated list of BCC (blind carbon copy) recipient email addresses.
- folder (TEXT): The email folder where the email is stored, such as "INBOX", "SPAM", "SENT", etc.
- flags (TEXT): Status flags such as "Seen", "Flagged", or other client-specific markers.
- has_attachments (INTEGER): A boolean True or False indicating whether the email contains attachments.

Please use this schema when generating SQL queries, ensuring field names and value formats are respected.
"""

first_state_prompt = """You are an intelligent assistant whose job is to decide whether the given relational database and query is it better to generate SQL query or use normal embedding search to get results from the database.
You will be provided:
1. A user query in natural language.
2. A SQL table schema with column names and types.

USER QUERY:
{query}

SQL TABLE SCHEMA:
{table_schema}

Your task is:
1. decide which method is better to get the results from the database.
2. If you think the query can be answered using SQL queries then return "sql_query".
3. If you think the query cannot be answered from SQL and it is better to use normal embedding search then return "embedding_search".
4. Do not give any explanation or additional text. Do not give response ```sql ``` and other characters.
5. Do not add any extra text or explanation. Just return "sql_query" or "embedding_search".
6. PLEASE NOT THAT NOT FOR EVEYQUESTION IT IS BETTER TO USE SQL QUERY. SOMETIMES IT IS BETTER TO USE NORMAL EMBEDDING SEARCH. PLEASE DO NOT ASSUME ANYTHING. ANS DO NOT THINK THAT WE CAN SEARCH IN CONTENT BY EXACT KEYWORD MATCHIN, IN THOSE CASES IT IS BETTER TO USE "embedding_search" INSTEAD OF SQL QUERY.

YOU TEND TO GIVE EVEYTIME THE SQL QUERY BUT SOMETIMES IT IS BETTER TO USE NORMAL EMBEDDING SEARCH. PLEASE DO NOT ASSUME ANYTHING.

FOR EXAMPLE IN QUESTIONS LIKE "Did i receive any emails from IMS India
" OR "Did i receive any emails from google about my account security?" IT IS BETTER TO USE Embedding query. Because we do not know the exact email or name of the sender. it is better to use embedding search instead of SQL query.

BUT FOR EXAMPLE IF THE QUESTIONS IS LIKE "What are in the emails from 3rd march 2021" OR "What are in the emails from 5th April 2021 between 10:00 AM to 11:00 AM" any emails which will require filtering based on given email table schema then it is better to use SQL query. Because we can filter the emails based on the given data.

In summary, only when we cannot do exact matching then use embedding search. But if we can do exact matching then use SQL query.
"""

sql_prompt = """You are an intelligent assistant whose job is to generate SQL query given the user query and the table schema.

You will be provided:
1. A user query in natural language.
2. A SQL table schema with column names and types.

USER QUERY:
{query}

SQL TABLE SCHEMA:
{table_schema}

Your taks is:
1. Generate a SQL SELECT query to get the results from the database.
2. The query should be a valid SQL query that can be executed on the given table schema.
3. The query should be in the format of a SQL SELECT statement.
4. The query should only include the columns that are needed to answer the user query.
5. The query should not include any extra information or explanation.
6. Do not add any extra text or explanation. Do not give response ```sql ``` and other characters.
7. Do not add any extra text or explanation. Just return the SQL query.

DO NOT ATTEMPT TO GUESS ANYTHING LIKE EMAIL ADDDRESS, SENDER NAME OF ANYTHING.
"""

import requests
import base64
import os

# def generate_response(prompt, image_path=None, model='meta/llama-4-maverick-17b-128e-instruct',
#                       stream=True, api_key='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'):
#     """
#     Generate a response from NVIDIA NIM API with optional image context.
    
#     Args:
#         prompt (str): The prompt to send to the model.
#         image_path (str, optional): Path to an image file (PNG/JPG). Defaults to None.
#         model (str): Model name. Defaults to Meta Llama 4 Maverick 17B.
#         stream (bool): Whether to stream the output or not. Defaults to True.
#         api_key (str): Your API key for NVIDIA NIM.
    
#     Returns:
#         str or generator: The model's response as a string or generator for streamed content.
#     """
#     invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"

#     # Optional Image Processing
#     image_b64 = ""
#     if image_path and os.path.isfile(image_path):
#         with open(image_path, "rb") as f:
#             image_b64 = base64.b64encode(f.read()).decode()
#         if len(image_b64) > 180_000:
#             raise ValueError("Image exceeds size limit. Use assets API for larger files.")

#     # Construct Content with Optional Image
#     content = prompt
#     if image_b64:
#         content += f' <img src="data:image/png;base64,{image_b64}" />'

#     # Prepare Headers and Payload
#     headers = {
#         "Authorization": f"Bearer {api_key}",
#         "Accept": "text/event-stream" if stream else "application/json"
#     }

#     payload = {
#         "model": model,
#         "messages": [{"role": "user", "content": content}],
#         "max_tokens": 512,
#         "temperature": 1.0,
#         "top_p": 1.0,
#         "stream": stream
#     }

#     # Make API Request
#     response = requests.post(invoke_url, headers=headers, json=payload)

#     # Handle Streaming or Non-Streaming
#     if stream:
#         def stream_response():
#             for line in response.iter_lines():
#                 if line:
#                     yield line.decode("utf-8")
#         return stream_response()
#     else:
#         return response.json()
import json

def generate_response(prompt, image_path=None, model='meta/llama-4-maverick-17b-128e-instruct',
                      stream=True, api_key='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'):
    invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"

    image_b64 = ""
    if image_path and os.path.isfile(image_path):
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()
        if len(image_b64) > 180_000:
            raise ValueError("Image exceeds size limit. Use assets API for larger files.")

    content = prompt
    if image_b64:
        content += f' <img src="data:image/png;base64,{image_b64}" />'

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "text/event-stream" if stream else "application/json"
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 512,
        "temperature": 1.0,
        "top_p": 1.0,
        "stream": stream
    }

    response = requests.post(invoke_url, headers=headers, json=payload, stream=stream)

    if stream:
        def stream_response():
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode("utf-8").strip()
                    if decoded_line.startswith("data: "):
                        try:
                            data = json.loads(decoded_line[len("data: "):])
                            content_chunk = data.get("choices", [{}])[0].get("delta", {}).get("content")
                            if content_chunk:
                                yield content_chunk
                        except json.JSONDecodeError:
                            continue
        return stream_response()
    else:
        return response.json()



def execute_sql_query(sql_query: str, db_path: str = r'C:\Personal\RAGEmail\emails.db'):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()
        column_names = [description[0] for description in cursor.description]
        conn.close()
        return [dict(zip(column_names, row)) for row in results]
    except sqlite3.Error as e:
        print(f"SQL error: {e}")
        return []


def format_docs_for_prompt(docs):
    formatted_docs = ""
    for i, doc in enumerate(docs):
        sender = doc.metadata.get("from_name", "Unknown")
        subject = doc.metadata.get("subject", "No subject")
        date = doc.metadata.get("sent_date", "Unknown date")
        formatted_docs += f"Document {i+1}:\nSubject: {subject}\nFrom: {sender}\nDate: {date}\nContent: {doc.page_content}\n\n"
    return formatted_docs


def process_query(user_query):
    answer_generation_prompt = """Your are an intelligent assistant whose job is to answer the user question using the given context.
    
    You will be provided:
    1. A user question in natural language.
    2. A context which is email

    USER QUESTION:
    {query}

    CONTEXT:
    {context}

    Instructions:
    1. Answer the user question using only the information provided in the context.
    2. Do not add any extra information or explanation or suggestions.
    3. Answer in the most helpful and complete way possible.
    4. answer in a nice format
    
    
    RETURN ANSWER IN CORRECT FORMAT WITH APPROPRIATE NEW LINE AND PARAGRAPHS.
    """
    # Prepare decision prompt
    decision_prompt = first_state_prompt.format(query=user_query, table_schema=table_schema)
    decision_result = generate_response(prompt=decision_prompt, stream=False, model="meta/llama-3.3-70b-instruct")

    if "sql_query" in decision_result["choices"][0]["message"]["content"]:
        print("SQL query generation")
        sql_query_prompt = sql_prompt.format(query=user_query, table_schema=table_schema)
        sql_response = generate_response(prompt=sql_query_prompt, stream=False)
        sql_query_text = sql_response["choices"][0]["message"]["content"]
        results = execute_sql_query(sql_query_text)
        source_emails = [
        {
            "message_id": row.get("message_id"),
            "subject": row.get("subject"),
            "from_email_id": row.get("from_email_id"),
            "to_email_id": row.get("to_email_id"),
            "sent_date": row.get("sent_date"),
            "body_preview": row.get("body_plain", "")[:20]  # preview of body
        } for row in results[:10]
        ]
        final_prompt = answer_generation_prompt.format(query=user_query, context=results)
    
        answer =  generate_response(prompt=final_prompt, stream=False, model="meta/llama-3.3-70b-instruct")
        return answer["choices"][0]["message"]["content"], source_emails

    elif "embedding_search" in decision_result["choices"][0]["message"]["content"]:
        print("Embedding search")
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 50})
        docs = retriever.get_relevant_documents(user_query)
        formatted_docs = format_docs_for_prompt(docs)
        final_prompt = answer_generation_prompt.format(query=user_query, context=formatted_docs)
        

        source_emails = [
            {
                "message_id": doc.metadata.get("message_id"),
                "subject": doc.metadata.get("subject"),
                "from_email_id": doc.metadata.get("from_email_id"),
                "to_email_id": doc.metadata.get("to_email_id"),
                "sent_date": doc.metadata.get("sent_date"),
                "body_preview": doc.page_content[:20]
            } for doc in docs[:10]  # Limit to first 10 documents
        ]
        answer =  generate_response(prompt=final_prompt, stream=False, model="meta/llama-3.3-70b-instruct")
        return answer["choices"][0]["message"]["content"], source_emails
    


    else:
        return {"error": "Model could not determine the retrieval method."}
