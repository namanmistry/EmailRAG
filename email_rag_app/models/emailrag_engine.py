import json
import sqlite3
from langchain.vectorstores import Chroma


# --- Static Schema and Prompts ---
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

# --- Helper Functions ---
def generate_response(prompt, image_path=None, model='meta/llama-4-maverick-17b-128e-instruct', stream=True, api_key=None):
    import requests, base64
    invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "text/event-stream" if stream else "application/json"
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 1.0,
        "top_p": 1.0,
        "stream": stream
    }

    response = requests.post(invoke_url, headers=headers, json=payload, stream=stream)

    if stream:
        def stream_response():
            for line in response.iter_lines():
                if line and line.startswith(b"data: "):
                    try:
                        data = json.loads(line[len(b"data: "):].decode("utf-8"))
                        chunk = data.get("choices", [{}])[0].get("delta", {}).get("content")
                        if chunk:
                            yield chunk
                    except json.JSONDecodeError:
                        continue
        return stream_response()
    else:
        return response.json()

def execute_sql_query(sql_query, db_path='emails.db'):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        conn.close()
        return [dict(zip(columns, row)) for row in rows]
    except sqlite3.Error as e:
        print(f"SQL error: {e}")
        return []

# --- Core Engine ---
class EmailRAGEngine:
    def __init__(self,embedding_model, db_path='emails.db', chroma_path='./email_chroma_db'):
        self.db_path = db_path
        self.embedding_model = embedding_model
        self.vector_store = Chroma(persist_directory=chroma_path, embedding_function=self.embedding_model)

    def decide_strategy(self, query, api_key):
        prompt = first_state_prompt.format(query=query, table_schema=table_schema)
        result = generate_response(prompt=prompt, stream=False, model="meta/llama-3.3-70b-instruct", api_key=api_key)
        return result["choices"][0]["message"]["content"].strip()

    def generate_sql(self, query, api_key):
        prompt = sql_prompt.format(query=query, table_schema=table_schema)
        result = generate_response(prompt=prompt, stream=False, model="meta/llama-3.3-70b-instruct", api_key=api_key)
        return result["choices"][0]["message"]["content"].strip()

    def process_query(self, query, api_key):
        strategy = self.decide_strategy(query, api_key)

        if strategy == "sql_query":
            sql = self.generate_sql(query, api_key)
            results = execute_sql_query(sql, db_path=self.db_path)
            preview = [
                {
                    "message_id": r.get("message_id"),
                    "subject": r.get("subject"),
                    "from_email_id": r.get("from_email_id"),
                    "to_email_id": r.get("to_email_id"),
                    "sent_date": r.get("sent_date"),
                    "body_preview": r.get("body_plain", "")[:20]
                } for r in results[:10]
            ]
            context = json.dumps(results, indent=2)

        elif strategy == "embedding_search":
            docs = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 50}).get_relevant_documents(query)
            preview = [
                {
                    "message_id": d.metadata.get("message_id"),
                    "subject": d.metadata.get("subject"),
                    "from_email_id": d.metadata.get("from_email_id"),
                    "to_email_id": d.metadata.get("to_email_id"),
                    "sent_date": d.metadata.get("sent_date"),
                    "body_preview": d.page_content[:20]
                } for d in docs[:10]
            ]
            context = docs

        else:
            return {"error": "Model could not determine the retrieval method."}, []

        final_prompt = answer_generation_prompt.format(query=query, context=context)
        answer = generate_response(prompt=final_prompt, stream=False, model="meta/llama-3.3-70b-instruct", api_key=api_key)
        return answer["choices"][0]["message"]["content"], preview