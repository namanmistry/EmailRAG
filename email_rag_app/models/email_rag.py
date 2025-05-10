import os
import re
import json
import sqlite3
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
import requests
# print("About to load model...", flush=True)
# try:
#     embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
#     print("Embedding model loaded successfully.", flush=True)
# except Exception as e:
#     print(f"Error loading embedding model: {e}", flush=True)

from langchain.schema import BaseRetriever, Document
from typing import List, Dict, Any
from pydantic import Field
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
import operator

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2")

class BalancedMetadataRetriever(BaseRetriever):
    vector_store: Any = Field(description="Vector store for document retrieval")
    top_k_per_value: int = Field(default=1)
    embedding_model: Any = embedding_model
    class Config:
        arbitrary_types_allowed = True
    def matches_filter_and(self, metadata: dict, filters: dict) -> bool:
        ops = {
            "$eq": operator.eq,
            "$ne": operator.ne,
            "$gt": operator.gt,
            "$gte": operator.ge,
            "$lt": operator.lt,
            "$lte": operator.le,
            "$in": lambda a, b: a in b,
            "$nin": lambda a, b: a not in b,
        }

        for key, condition in filters.items():
            if key not in metadata:
                return False  # Key missing = fail

            actual_value = metadata[key]
            if not isinstance(condition, dict):
                condition = {"$eq": condition}  # fallback

            matched = False

            for op_key, expected_value in condition.items():
                if op_key in ops:
                    try:
                        # Normalize strings
                        if isinstance(actual_value, str) and isinstance(expected_value, str):
                            actual_value = actual_value.lower().strip()
                            expected_value = expected_value.lower().strip()

                        # Print for debugging
                       

                        if ops[op_key](actual_value, expected_value):
                            matched = True  # condition passed
                        else:
                            matched = False
                            break  # no need to check other ops on this key

                    except:
                        return False

            if not matched:
                return False  # this key did not satisfy any condition

        return True  # all keys matched their conditions

    def matches_filter_or(self,metadata: dict, filters: dict) -> bool:
        ops = {
            "$eq": operator.eq,
            "$ne": operator.ne,
            "$gt": operator.gt,
            "$gte": operator.ge,
            "$lt": operator.lt,
            "$lte": operator.le,
            "$in": lambda a, b: a in b,
            "$nin": lambda a, b: a not in b,
        }

        for key, condition in filters.items():
            if key not in metadata:
                continue

            actual_value = metadata[key]
            if not isinstance(condition, dict):
                condition = {"$eq": condition}  # fallback

            for op_key, expected_value in condition.items():
                if op_key in ops:
                    try:
                        # Normalize string values for comparison
                        if isinstance(actual_value, str) and isinstance(expected_value, str):
                            actual_value = actual_value.lower().strip()
                            expected_value = expected_value.lower().strip()
  
                        if ops[op_key](actual_value, expected_value):
                            return True  # OR logic: return on first match
                    except:
                        continue
        return False  # No condition matched
    def _get_relevant_documents(self, query: str, metadata: Dict[str, Any]) -> List[Document]:
        # ✅ Step 1: Load only metadata + IDs (not embeddings!)
        raw = self.vector_store._collection.get(include=["metadatas"])
        all_metadatas = raw["metadatas"]
        all_ids = raw["ids"]

        
        filtered_ids = [id for d, id in zip(all_metadatas, all_ids) if self.matches_filter_and(d, metadata)]

        if not filtered_ids:
            return []


        # ✅ Step 3: Load docs/embeddings only for filtered IDs
        result = self.vector_store.get(ids=filtered_ids, include=["documents", "metadatas", "embeddings"])
        docs = result["documents"]
        metas = result["metadatas"]
        embs = result["embeddings"]

        # ✅ Step 4: Embed the query
        query_embedding = self.embedding_model.embed_query(query)

        def cosine_similarity(a, b):
            a, b = np.array(a), np.array(b)
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

        # ✅ Step 5: Rank filtered docs by similarity
        doc_sim_pairs = [
            (Document(page_content=doc, metadata=meta), cosine_similarity(query_embedding, emb))
            for doc, meta, emb in zip(docs, metas, embs)
        ]
        doc_sim_pairs.sort(key=lambda x: x[1], reverse=True)

        # ✅ Return top-k
        final_docs = [doc for doc, _ in doc_sim_pairs[:5]]
        
        if len(final_docs) == 0:
            ret = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 50})
            final_docs = ret.get_relevant_documents(query)
        return [doc for doc, _ in doc_sim_pairs[:50]]

class EmailRAG:
    def __init__(self):
        # Initialize embedding and load vector store if exists
        print("i have entered")
       
        
        persist_dir = r"C:\Personal\RAGEmail\email_chroma_db_new_2000"
        print("embedding mode loaded")
        if os.path.exists(persist_dir) and os.listdir(persist_dir):
            print("Loading existing vector database...")
            self.vector_store = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
        else:
            raise ValueError("Vector store not found. Please generate embeddings first.")
        print("Vector store loaded successfully.")
        # Metadata extraction LLM
        self.metadata_llm = ChatOllama(
            model="gemma3:4b-it-qat",
            base_url="http://localhost:11434",
            temperature=0.9
        )
        print("Metadata LLM loaded successfully.")
        # Metadata prompt template
        self.metadata_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("human", "{question}")
        ])
        self.chain = self.metadata_prompt | self.metadata_llm
        # print("Initializedg EmailRAG...")
        self.custom_retriever = BalancedMetadataRetriever(vector_store=self.vector_store, top_k_per_value=5)

        print("metadata prompt template loaded successfully.")

    def metadata_filtering(self, query):
        metadata = {"from_name": "sender's name", "from_email_id": "sender's email address", "cc_addresses": "email addresses in cc", "bcc_addresses": "email addresses in bcc", "folder": "email folder like INBOX, SPAM (NOT ANY OTHER)", "has_attachments": "whether the email has attachments", "message_id": "unique identifier for the email",
            "date": "date the email was sent in iso format","time": "time the email was sent in iso format","year":"year the email was sent","month": "month the email was sent"}

        # question = """YOUR JOB IS THE EXTRACT KEYWORDS ACCORDING TO BELOW METADATA IF THEY EXISTS IN USER QUERY.

        # I WILL PROVIDE YOU THE USER QUERY AND THE METADATA. EXTRACT THE KEYWORDS FROM THE USER QUERY WHICH CAN BE USED TO FILTER AVAILABLE METADATA AS A COMMA SEPARATED LIST. 

        # RETURN YOUR ANSWER IN THE FOLLOWING FORMAT:
        # {{\"key_name_1\": [\"value_1\", \"value_2\", ...], \"key_name_2\": [\"value_1\", \"value_2\", ...], ...}}

        # PLEASE BE CAREFUL WHEN ASSIGNING THE VALUES TO KEYS YOU MUST ASSIGN THE VALUES TO THE CORRECT KEYS.

        # DO NOT GIVE ANY EXPLANATION OR ADDITIONAL TEXT. DO NOT GIVE response ```json ``` and other characters.

        # HERE IS USER QUERY:
        # {query}

        # HERE IS METADATA:
        # {metadata}
        # """
        question = """You are an intelligent assistant that extracts ChromaDB-compatible metadata filters from a user query.

        You will be provided:
        1. A user query in natural language.
        2. A list of metadata fields and their meanings.

        Your task is to analyze the query and extract all relevant filters using the correct field names and appropriate comparison operators supported by ChromaDB.

        Return your answer **STRICTLY** in this JSON format no matter what (without explanation or comments):
        {{
        "field_name1": {{"$operator": value_or_list}},
        "field_name2": {{"$operator": value_or_list}},
        ...
        }}

        Supported ChromaDB operators:
        - "$eq": equals
        - "$ne": not equal to
        - "$gt": greater than (use for date/time/numbers)
        - "$gte": greater than or equal
        - "$lt": less than
        - "$lte": less than or equal
        - "$in": one of these values
        - "$nin": not in these values

        <STRICT INSTRUCTIONA>
        1.DO NOT wrap the response in triple backticks, markdown, or add the word 'json'. Return only the raw JSON object. YOU MUST ANSWER IN THE FORMAT SPECIFIED ABOVE.
        2.IF YOU THINK QUERY IS GENERAL AND NO METADATA FIELDS CAN BE EXTRACTED THEN ONLY RETURN AN EMPTY JSON OBJECT. BUT IN GENERAL YOU MUST TRY TO EXTRACT THE METADATA FIELDS EVEN IF ONLY ONE IS RELEVANT.
        3. DO NOT ASSUME ANYTHING ABOUT date, year, sender, folder anything. DO NOT EXTRACT ANYTHING THAT IS NOT IN THE USER QUERY. DO NOT ADD ANYTHING EXTRA. DO NOT GIVE EXAMPLES OR ADDITIONAL TEXT.
        </STRICT INSTRUCTIONA>
        
        METADATA FIELDS:
        {metadata}

        USER QUERY:
        {query}
        """


        question = question.format(query=query, metadata=metadata)
        response = self.chain.invoke({"question": question})
        print("response", response.content)
        cleaned = re.sub(r'(?s)<think>.*?</think>', '', response.content)
        cleaned = re.sub(r"```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip("` \n")
        json_data = json.loads(cleaned)
        new_json_data = {}
        for key, value in json_data.items():
            if key in metadata.keys():
                new_json_data[key] = value
        return new_json_data
    
    # def metadata_filtering(self, query):
    #     metadata = {

    #         "folder": "email folder like INBOX, SPAM (NOT ANY OTHER)",
    #         "has_attachments": "whether the email has attachments",
    #         "message_id": "unique identifier for the email",
    #         "date": "date the email was sent in iso format",
    #         "time": "time the email was sent in iso format",
    #         "year": "year the email was sent",
    #         "month": "month the email was sent"
    #     }

    #     question = """You are an intelligent assistant that extracts ChromaDB-compatible metadata filters from a user query.

    #     You will be provided:
    #     1. A user query in natural language.
    #     2. A list of metadata fields and their meanings.

    #     Your task is to analyze the query and extract all relevant filters using the correct field names and appropriate comparison operators supported by ChromaDB.

    #     Return your answer **STRICTLY** in this JSON format no matter what (without explanation or comments):
    #     {{
    #     "field_name1": {{"$operator": value_or_list}},
    #     "field_name2": {{"$operator": value_or_list}},
    #     ...
    #     }}

    #     Supported ChromaDB operators:
    #     - "$eq": equals
    #     - "$ne": not equal to
    #     - "$gt": greater than (use for date/time/numbers)
    #     - "$gte": greater than or equal
    #     - "$lt": less than
    #     - "$lte": less than or equal
    #     - "$in": one of these values
    #     - "$nin": not in these values

    #     <STRICT INSTRUCTIONA>
    #     1.DO NOT wrap the response in triple backticks, markdown, or add the word 'json'. Return only the raw JSON object. YOU MUST ANSWER IN THE FORMAT SPECIFIED ABOVE.
    #     2.IF YOU THINK QUERY IS GENERAL AND NO METADATA FIELDS CAN BE EXTRACTED THEN ONLY RETURN AN EMPTY JSON OBJECT. BUT IN GENERAL YOU MUST TRY TO EXTRACT THE METADATA FIELDS EVEN IF ONLY ONE IS RELEVANT.
    #     3. DO NOT ASSUME ANYTHING ABOUT date, year, sender, folder anything. DO NOT EXTRACT ANYTHING THAT IS NOT IN THE USER QUERY. DO NOT ADD ANYTHING EXTRA. DO NOT GIVE EXAMPLES OR ADDITIONAL TEXT.
    #     </STRICT INSTRUCTIONA>
        
    #     METADATA FIELDS:
    #     {metadata}

    #     USER QUERY:
    #     {query}
    #     """

    #     prompt = question.format(query=query, metadata=metadata)

    #     # NIM API endpoint and model
    #     LLM_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"
    #     NIM_MODEL = "meta/llama-4-scout-17b-16e-instruct"  # or another suitable model

    #     headers = {
    #         "Authorization": "Bearer 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",
    #         "Content-Type": "application/json"
    #     }

    #     payload = {
    #         "model": NIM_MODEL,
    #         "messages": [
    #             {"role": "user", "content": prompt}
    #         ],
    #         "temperature": 0.2,
    #         "top_p": 0.7,
    #         "max_tokens": 512,
    #         "stream": False
    #     }

    #     response = requests.post(LLM_ENDPOINT, headers=headers, json=payload)
    #     response.raise_for_status()
    #     result = response.json()
    #     content = result['choices'][0]['message']['content']

    #     # Clean up and parse JSON as before
    #     cleaned = re.sub(r'(?s)<think>.*?</think>', '', content)
    #     json_data = json.loads(cleaned)
    #     new_json_data = {}
    #     for key, value in json_data.items():
    #         if key in metadata.keys():
    #             new_json_data[key] = value
    #     return new_json_data

    def format_docs_for_prompt(self, docs):
        formatted = ""
        for i, doc in enumerate(docs):
            formatted += f"Document {i+1}:\nSubject: {doc.metadata.get('subject', 'No subject')}\nFrom: {doc.metadata.get('from_name', 'Unknown')}\nDate: {doc.metadata.get('sent_date', 'Unknown date')}\nContent: {doc.page_content}\n\n"
        return formatted

    def strip_metadata_from_content(self, doc, metadata_keys):
        pattern = re.compile(rf"^({'|'.join(re.escape(key) for key in metadata_keys)}):.*(?:\n|$)", re.MULTILINE)
        doc.page_content = pattern.sub('', doc.page_content).lstrip('\n')
        return doc
    def format_docs_for_prompt(self, docs):
        formatted_docs = ""
        for i, doc in enumerate(docs):
            sender = doc.metadata.get("from_name", "Unknown")
            subject = doc.metadata.get("subject", "No subject")
            date = doc.metadata.get("sent_date", "Unknown date")
            
            formatted_docs += f"Document {i+1}:\n"
            formatted_docs += f"Subject: {subject}\n"
            formatted_docs += f"From: {sender}\n"
            formatted_docs += f"Date: {date}\n"
            formatted_docs += f"Content: {doc.page_content}\n\n"
        
        return formatted_docs
    
    def answer_question(self, question, top_k=5):
        # Metadata filtering
        # metadata_filter = self.metadata_filtering(question)
        # print(f"Extracted Metadata Filter: {metadata_filter}")

        # # Retrieve documents
        # docs = self.custom_retriever._get_relevant_documents(question, metadata=metadata_filter)
        retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        formatted_docs = retriever.get_relevant_documents(question)
        formatted_docs = self.format_docs_for_prompt(formatted_docs)
        if not formatted_docs:
            yield "No relevant documents found."
            return

        # m_keys = list(docs[0].metadata.keys())
        # for skip_key in ["subject", "from_name", "from_email_id", "date"]:
        #     if skip_key in m_keys:
        #         m_keys.remove(skip_key)

        # cleaned_docs = [self.strip_metadata_from_content(doc, m_keys) for doc in docs]
        # formatted_docs = self.format_docs_for_prompt(cleaned_docs)

        # RAG Prompt
        rag_prompt_template = """
            <s>[INST] <<SYS>>
            You are a useful AI assistant who will answer the user's question using the provided context.

            <MUST FOLLOW INSTRUCTIONS>
            1. Answer ONLY the asked question.
            2. Be concise and precise.
            3. Do not add extra information or examples.
            </MUST FOLLOW INSTRUCTIONS>
            <</SYS>>

            USER QUESTION:
            {query}

            CONTEXT:
            {formatted_docs}

            NOW GIVE ANSWER.
            [/INST]
            """

        prompt = rag_prompt_template.format(query=question, formatted_docs=formatted_docs)

        # Initialize the Ollama LLM
        ollama_llm = Ollama(
            model="llama3.2:3b-instruct-q2_K",
            base_url="http://localhost:11434",
            temperature=0.9,
            num_thread=12
        )

        # Stream response directly from the LLM
        for token in ollama_llm.stream(prompt):
            print(token, end='', flush=True)
            yield token
        


    # def answer_question(self, question, top_k=5):
    #     metadata_filter = self.metadata_filtering(question)
    #     print(f"Extracted Metadata Filter: {metadata_filter}")
    #     docs = self.custom_retriever._get_relevant_documents(question, metadata=metadata_filter)
    #     # print(f"Extracted Metadata Filter: {metadata_filter}")
    #     # retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 50})
    #     # formatted_docs = retriever.get_relevant_documents(question)
    #     formatted_docs = self.format_docs_for_prompt(docs)
    #     print("formatted_docs", formatted_docs)
    #     if not formatted_docs:
    #         yield "No relevant documents found."
    #         return

    #     rag_prompt_template = """
    #         <s>[INST] <<SYS>>
    #         You are a useful AI assistant who will answer the user's question using the provided context.

    #         <MUST FOLLOW INSTRUCTIONS>
    #         1. Answer ONLY the asked question.
    #         2. Be concise and precise.
    #         3. Do not add extra information or examples.
    #         4. answer only based on the context provided.
    #         </MUST FOLLOW INSTRUCTIONS>
    #         <</SYS>>

    #         USER QUESTION:
    #         {query}

    #         CONTEXT:
    #         {formatted_docs}

    #         NOW GIVE ANSWER.
    #         [/INST]
    #         """
    #     prompt = rag_prompt_template.format(query=question, formatted_docs=formatted_docs)

    #     invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
    #     stream = True

    #     headers = {
    #         "Authorization": "Bearer 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",
    #         "Accept": "text/event-stream" if stream else "application/json"
    #     }

    #     payload = {
    #         "model": "meta/llama-4-scout-17b-16e-instruct",
    #         "messages": [
    #             {
    #                 "role": "user",
    #                 "content": prompt
    #             }
    #         ],
    #         "max_tokens": 512,
    #         "temperature": 0.9,
    #         "top_p": 1.0,
    #         "stream": stream
    #     }

    #     response = requests.post(invoke_url, headers=headers, json=payload, stream=stream)
    #     if stream:
    #         for line in response.iter_lines():
    #             if line:
    #                 decoded = line.decode("utf-8")
    #                 # NVIDIA NIM streams JSON lines starting with "data:"
    #                 if decoded.startswith("data:"):
    #                     # Remove "data:" prefix and parse JSON
    #                     data = decoded[len("data:"):].strip()
    #                     if data == "[DONE]":
    #                         break
    #                     try:
    #                         import json
    #                         chunk = json.loads(data)
    #                         content = chunk["choices"][0]["delta"].get("content", "")
    #                         print(content, end='', flush=True)
    #                         yield content
    #                     except Exception:
    #                         continue
    #     else:
    #         result = response.json()
    #         content = result["choices"][0]["message"]["content"]
    #         print(content)
    #         yield content

# obj = EmailRAG()
# print("EmailRAG object created successfully.")