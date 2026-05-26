import os
import faiss
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from openai import OpenAI

# =========================
# OPENAI API KEY
# =========================

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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"
)

# =========================
# BASE DIRECTORY
# =========================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# =========================
# PATH
# =========================

faiss_path = os.path.join(
    BASE_DIR,
    "models",
    "faiss_index.index"
)

data_path = os.path.join(
    BASE_DIR,
    "models",
    "data_obat.pkl"
)

# =========================
# LOAD EMBEDDING MODEL
# =========================

embedding_model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)

# =========================
# LOAD FAISS
# =========================

index = faiss.read_index(faiss_path)

# =========================
# LOAD DATAFRAME
# =========================

df = pd.read_pickle(data_path)

# =========================
# RETRIEVER
# =========================

def retrieve_documents(query, top_k=5):

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    )

    distances, indices = index.search(
        np.array(query_embedding, dtype=np.float32),
        top_k
    )

    results = []

    for idx in indices[0]:

        results.append({
            "nama_obat": df.iloc[idx]["name_master_erp"],
            "kategori": df.iloc[idx]["Golongan Obat"],
            "indikasi": df.iloc[idx]["Indikasi"],
            "harga": df.iloc[idx]["Harga"],
            "kontraindikasi": df.iloc[idx]["Kontraindikasi"],
            "efek_samping": df.iloc[idx]["Efek Samping"]
        })

    return results

# =========================
# AI AGENT
# =========================

def ask_ai(query):

    docs = retrieve_documents(query)

    context = ""

    for doc in docs:

        context += f"""
Nama Obat: {doc['nama_obat']}
Kategori: {doc['kategori']}
Indikasi: {doc['indikasi']}
Harga: {doc['harga']}
Kontraindikasi: {doc['kontraindikasi']}
Efek Samping: {doc['efek_samping']}

========================
"""

    prompt = f"""
Kamu adalah AI Agent Farmasi profesional.

Tugas kamu:
- Menjawab pertanyaan user berdasarkan data obat.
- Berikan jawaban singkat, jelas, dan aman.
- Jangan memberikan diagnosis medis.
- Jangan membuat informasi palsu.

DATA OBAT:
{context}

PERTANYAAN USER:
{query}

JAWABAN:
"""

    response = client.chat.completions.create(
        model="openai/gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "Kamu adalah AI Agent Farmasi profesional."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3
    )

    return response.choices[0].message.content
