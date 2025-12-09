# app.py
import os
import sqlite3
from datetime import datetime
from io import StringIO

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import altair as alt
from sqlalchemy import create_engine
import requests
from dotenv import load_dotenv

load_dotenv()  # untuk memuat GROQ_API_KEY dari .env jika ada

# ---------- konfigurasi DB ----------
DB_PATH = "data/transactions.db"
os.makedirs("data", exist_ok=True)
engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)

def init_db():
    with engine.connect() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            type TEXT NOT NULL,        -- "in" atau "out"
            category TEXT,
            amount REAL NOT NULL,
            note TEXT
        );
        """)
init_db()

# ---------- helper ----------
def insert_tx(date, tx_type, category, amount, note):
    df = pd.DataFrame([{
        "date": date,
        "type": tx_type,
        "category": category,
        "amount": float(amount),
        "note": note
    }])
    df.to_sql("transactions", engine, if_exists="append", index=False)

def load_transactions():
    return pd.read_sql("SELECT * FROM transactions ORDER BY date DESC, id DESC", engine)

def export_csv(df):
    return df.to_csv(index=False).encode("utf-8")

# ---------- UI ----------
st.set_page_config(page_title="Kas Masuk & Keluar", layout="wide")
st.title("📒 Aplikasi Kas — Masuk & Keluar")

# dua kolom: form input dan ringkasan
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Tambah transaksi")
    date = st.date_input("Tanggal", value=datetime.today())
    tx_type = st.selectbox("Tipe", ["in", "out"], format_func=lambda x: "Masuk" if x=="in" else "Keluar")
    category = st.text_input("Kategori (opsional)", value="")
    amount = st.number_input("Jumlah (Rp)", min_value=0.0, format="%.2f")
    note = st.text_area("Keterangan (opsional)", height=80)
    if st.button("Simpan transaksi"):
        insert_tx(date.isoformat(), tx_type, category, amount, note)
        st.success("Transaksi tersimpan ✅")

    st.markdown("---")
    st.header("Impor / Ekspor")
    uploaded = st.file_uploader("Impor CSV (kolom: date,type,category,amount,note)", type=["csv"])
    if uploaded:
        df_up = pd.read_csv(uploaded)
        expected = {"date","type","category","amount","note"}
        if expected.issubset(set(df_up.columns)):
            df_up.to_sql("transactions", engine, if_exists="append", index=False)
            st.success(f"{len(df_up)} transaksi diimpor")
        else:
            st.error("Format CSV salah. Pastikan kolom date,type,category,amount,note ada.")
    if st.button("Ekspor CSV semua transaksi"):
        df_all = load_transactions()
        csv_bytes = export_csv(df_all)
        st.download_button("Download CSV", data=csv_bytes, file_name="transactions.csv", mime="text/csv")

with col2:
    st.header("Ringkasan & Tabel")
    df = load_transactions()
    if df.empty:
        st.info("Belum ada transaksi. Tambah transaksi di panel kiri.")
    else:
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        # ringkasan
        total_masuk = df.loc[df['type']=='in', 'amount'].sum()
        total_keluar = df.loc[df['type']=='out', 'amount'].sum()
        saldo = total_masuk - total_keluar

        st.metric("Total Masuk (Rp)", f"{total_masuk:,.2f}")
        st.metric("Total Keluar (Rp)", f"{total_keluar:,.2f}")
        st.metric("Saldo (Rp)", f"{saldo:,.2f}")

        st.subheader("Filter dan Tabel")
        colf1, colf2, colf3 = st.columns([2,2,1])
        with colf1:
            date_start = st.date_input("Dari", value=df['date'].min().date())
        with colf2:
            date_end = st.date_input("Sampai", value=df['date'].max().date())
        with colf3:
            jenis = st.selectbox("Tipe", ["All", "Masuk", "Keluar"])

        mask = (df['date'].dt.date >= date_start) & (df['date'].dt.date <= date_end)
        if jenis != "All":
            mask &= (df['type'] == ('in' if jenis=="Masuk" else 'out'))
        df_filtered = df.loc[mask].copy()
        st.dataframe(df_filtered.sort_values(by='date', ascending=False), use_container_width=True)

        # Chart: trend harian (matplotlib)
        st.subheader("Trend Harian (Jumlah)")
        daily = df_filtered.groupby([df_filtered['date'].dt.date, 'type'])['amount'].sum().unstack(fill_value=0)
        if not daily.empty:
            fig, ax = plt.subplots(figsize=(8,3.5))
            daily.plot(ax=ax)
            ax.set_xlabel("Tanggal")
            ax.set_ylabel("Jumlah (Rp)")
            ax.grid(True, linestyle=':', linewidth=0.5)
            st.pyplot(fig)
        else:
            st.info("Tidak ada data untuk rentang tanggal yang dipilih.")

        # Pie chart kategori (altair)
        st.subheader("Distribusi per Kategori (Top 10)")
        cat = df_filtered.groupby('category')['amount'].sum().sort_values(ascending=False).head(10).reset_index()
        if not cat.empty:
            chart = alt.Chart(cat).mark_arc().encode(
                theta=alt.Theta(field="amount", type="quantitative"),
                color=alt.Color(field="category", type="nominal"),
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Belum ada kategori untuk ditampilkan.")

# ---------- AI / ANALISIS (opsional) ----------
st.markdown("---")
st.header("🔎 AI Insights (opsional)")
st.write("Gunakan Groq untuk menganalisis transaksi — mis. ringkasan, rekomendasi penghematan, atau pengelompokan kategori otomatis.")
col_ai1, col_ai2 = st.columns([2,1])
with col_ai1:
    enable_ai = st.checkbox("Aktifkan analisis Groq (butuh API key di env: GROQ_API_KEY)")
    if enable_ai:
        st.info("Pastikan variabel lingkungan GROQ_API_KEY sudah diset (atau buat file .env dengan GROQ_API_KEY=...)")

with col_ai2:
    if st.button("Jalankan Analisis AI (ringkasan 10 transaksi terakhir)"):
        if not enable_ai:
            st.warning("Centang 'Aktifkan analisis Groq' terlebih dahulu.")
        else:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                st.error("GROQ_API_KEY tidak ditemukan di environment.")
            else:
                # ambil 10 transaksi terakhir
                sample = df.head(10).to_dict(orient="records")
                prompt = "Berikan ringkasan singkat dalam bahasa Indonesia dari transaksi berikut, dan berikan 3 rekomendasi untuk mengurangi pengeluaran jika ada.\n\nTransaksi:\n"
                for t in sample:
                    prompt += f"- {t['date'].strftime('%Y-%m-%d')}: {'Masuk' if t['type']=='in' else 'Keluar'} Rp{t['amount']:,.2f} | Kategori: {t.get('category','-')} | Keterangan: {t.get('note','-')}\n"
                st.markdown("**Prompt yang dikirim ke model:**")
                st.code(prompt[:1000] + ("..." if len(prompt)>1000 else ""))

                # --- CARA A: gunakan Groq SDK (jika diinstall)
                try:
                    from groq import Client as GroqClient
                    client = GroqClient(api_key=api_key)
                    # contoh pemanggilan sederhana (sesuaikan model berdasarkan akun Anda)
                    resp = client.responses.create(
                        model="llama-3.3-70b-versatile",
                        input=prompt,
                        max_output_tokens=400
                    )
                    # struktur respons bergantung pada SDK; ambil text secara defensif
                    text = ""
                    if hasattr(resp, "output_text"):
                        text = resp.output_text
                    else:
                        # fallback: cari field choices atau output[0].content
                        try:
                            text = resp.get("output", [{}])[0].get("content", [{}])[0].get("text", "")
                        except Exception:
                            text = str(resp)
                    st.subheader("Hasil Analisis (Groq SDK)")
                    st.write(text)
                    st.success("Selesai (Groq SDK).")
                except Exception as e_sdk:
                    st.warning("Groq SDK tidak tersedia atau panggilan SDK gagal. Mencoba metode HTTP fallback. (Pesan: %s)" % str(e_sdk))

                    # --- CARA B: OpenAI-compatible HTTP POST ke endpoint Groq
                    try:
                        url = "https://api.groq.com/openai/v1/chat/completions"
                        headers = {
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json"
                        }
                        payload = {
                            "model": "llama-3.3-70b-versatile",
                            "messages": [{"role":"user","content": prompt}],
                            "max_tokens": 400,
                            "temperature": 0.2
                        }
                        r = requests.post(url, json=payload, headers=headers, timeout=30)
                        r.raise_for_status()
                        j = r.json()
                        # Ambil text hasil (OpenAI-style)
                        text = ""
                        if "choices" in j and len(j["choices"])>0:
                            text = j["choices"][0].get("message", {}).get("content", "")
                        else:
                            text = str(j)
                        st.subheader("Hasil Analisis (HTTP ke Groq OpenAI-compatible endpoint)")
                        st.write(text)
                        st.success("Selesai (HTTP).")
                    except Exception as e_http:
                        st.error("Analisis AI gagal: %s" % str(e_http))

st.markdown("---")
st.caption("App ini menyimpan data di file SQLite pada folder `data/transactions.db` di repo. Jangan commit API key ke GitHub — gunakan Secrets / .env lokal.")
