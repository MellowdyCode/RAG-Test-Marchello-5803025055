from pathlib import Path
from pypdf import PdfReader

def baca_pdf(path):
    reader = PdfReader(path)
    return "\n".join((p.extract_text() or "") for p in reader.pages)

def potong(teks, ukuran=800, tumpang=100):
    """Pemotongan ukuran tetap dengan overlap (versi paling sederhana)."""
    hasil, mulai = [], 0
    while mulai < len(teks):
        hasil.append(teks[mulai:mulai + ukuran])
        mulai += ukuran - tumpang
    return [h for h in hasil if h.strip()]

import chromadb
from chromadb.utils import embedding_functions

# model embedding multilingual (jalan lokal, tanpa API key)
emb = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2")

client = chromadb.PersistentClient(path="./chroma_db")   # tersimpan di folder
koleksi = client.get_or_create_collection(
    name="pedoman", embedding_function=emb,
    metadata={"hnsw:space": "cosine"})                    # ukuran jarak: cosine

for pdf in Path("dokumen").glob("*.pdf"):
    chunks = potong(baca_pdf(pdf))
    koleksi.upsert(
        ids=[f"{pdf.stem}-{i}" for i in range(len(chunks))],
        documents=chunks,
        metadatas=[{"sumber": pdf.name, "chunk": i} for i in range(len(chunks))],
    )
    print(pdf.name, "→", len(chunks), "chunk")

hasil = koleksi.query(query_texts=["Apa syarat cuti akademik?"], n_results=5)

for doc, meta, jarak in zip(hasil["documents"][0], hasil["metadatas"][0], hasil["distances"][0]):
    print(f"[{meta['sumber']} · chunk {meta['chunk']}]  jarak={jarak:.3f}")
    print(doc, "\n")  

# (pertanyaan, kata/frasa yang HARUS ada di chunk jawaban yang benar)
UJI = [
    ("Berapa persen biaya yang harus dibayar mahasiswa yang cuti akademik?", "lima puluh persen"),
    ("Berapa semester minimal harus sudah dijalani sebelum boleh cuti?", "minimal sudah mengikuti perkuliahan"),
    ("Apakah mahasiswa yang cuti harus melunasi tunggakan?", "tidak mempunyai tunggakan"),
    ("Apakah SKS yang sudah diperoleh hangus saat cuti?", "tetap diakui"),
    ("Apakah masa cuti dihitung sebagai masa studi?", "tidak diperhitungkan sebagai masa studi"),
    ("Kegiatan apa yang dilarang selama cuti?", "kegiatan kemahasiswaan"),
    ("Kapan harus mengajukan permohonan aktif kembali setelah cuti?", "bulan sebelum semester dimulai"),
    ("Peraturan rektor nomor berapa yang dinyatakan tidak berlaku lagi?", "857"),
    ("Paling lambat kapan permohonan cuti diajukan sebelum semester?", "dua minggu"),
    ("Apakah mahasiswa semester awal boleh langsung cuti?", "semester pertama"),
    ("Apakah penerima beasiswa boleh mengambil cuti?", "beasiswa"),
    ("Apa kewajiban mahasiswa cuti terkait registrasi?", "registrasi ulang"),
    ("Apa yang dimaksud dengan cuti studi?", "berhenti studi sementara"),
    ("Permohonan cuti studi diajukan ke Rektor melalui siapa?", "melalui wakil"),
    ("Bagaimana alur pengajuan sampai terbit surat keterangan?", "surat keterangan"),
]

def hit_at_k(uji, k=3):
    benar = 0
    for pertanyaan, kunci in uji:
        docs = koleksi.query(query_texts=[pertanyaan], n_results=k)["documents"][0]
        if any(kunci.lower() in d.lower() for d in docs):
            benar += 1
    return benar / len(uji)

for k in (1, 3, 5):
    print(f"Hit@{k} = {hit_at_k(UJI, k):.2f}")