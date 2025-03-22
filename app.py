import openai
from fastapi import FastAPI, HTTPException
from openai import OpenAI
import os
import ollama
import psycopg2
import numpy as np
from pydantic import BaseModel
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv

load_dotenv(override=True, verbose=True)
app = FastAPI()

conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port="5432"
)
register_vector(conn)

OLLAMA_URL = "http://localhost:11434/api/embed"
openai_api_key = os.getenv("OPENAI_API_KEY")

openai_client = OpenAI(api_key=openai_api_key)

class Note(BaseModel):
    title: str
    text: str

class Message(BaseModel):
    role: str
    message: str

class ChatRequest(BaseModel):
    messages: list[Message]

def get_embeddings(text: str):
        response = ollama.embed(model="nomic-embed-text", input=text)
        return response["embeddings"]

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/add_note")
async def add_note(note: Note):
    try:
        embedding = get_embeddings(note.text)
        with conn.cursor() as cursor:
            cursor.execute("INSERT INTO notes (title, text, embedding) VALUES (%s, %s, %s)",
                           (note.title, note.text, np.array(embedding[0])))
            conn.commit()
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(f"There was an error adding notes: {e}")

@app.get("/search_notes")
async def search_notes(query: str):
    try:
        embedding = get_embeddings(query)
        with conn.cursor() as cursor:
            cursor.execute("SELECT title, text FROM notes ORDER BY embedding <=> %s LIMIT 5", (np.array(embedding[0]),))
            results = cursor.fetchall()
        return {
            "total": len(results),
            "results": [
                {"title": row[0], "text": row[1]} for row in results
            ]
        }
    except Exception as e:
        raise HTTPException(f"There was an error searching notes: {e}")

@app.get("/all_notes")
async def all_notes():
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT title, text FROM notes")
            results = cursor.fetchall()

        notes = []
        for row in results:
            title, text = row

            # if isinstance(embedding, np.ndarray):
            #     print("this is a ndarray")
            #     embedding = embedding.tolist()
            #
            # elif isinstance(embedding, np.float32):
            #     print("this is a float32")
            #     embedding = float(embedding)

            notes.append({"title": title, "text": text})

        return {"notes": notes}
    except Exception as e:
        raise HTTPException(f"There was an error getting all notes: {e}")

# @app.post("/chat_api")
# async def chat_api(message: Message):
#     prompt = {
#         "role": message.role,
#         "message": message.message,
#     }
