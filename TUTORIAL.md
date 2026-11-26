# TUTORIAL — dari nol sampai ship: Policy Q&A RAG API di Docker + Bedrock

Halo Bryan. Ini catatan kelas lengkap buat proyek **PolicyLens** (repo:
`DOCKER-AWS-TRIAL`). Semua angka di sini **hasil ukur beneran** di mesin kamu
(Apple M5 Pro, macOS 27.2.1, Docker 29.8.2, Compose 5.6.1, Python 3.13.18), bukan
angka karangan. Kalau ada yang belum bisa diverifikasi (terutama soal Bedrock), aku
tulis jelas "belum diverifikasi" dan cara ceknya.

Cara pakai tutorial ini: baca satu milestone, jalankan command-nya, lalu jawab 3
check question di akhir milestone sebelum lanjut. Jangan skip Milestone 3–4 kalau
kamu mau bisa jawab pertanyaan interview soal Docker.

---

## 0. Environment check (yang kita temukan)

```bash
uname -a && sw_vers
docker --version && docker compose version
python3 --version
aws --version
```

Hasil di mesin kamu:

| Tool | Versi | Catatan |
|---|---|---|
| OS | macOS 27.2.1, Apple M5 Pro, 48 GB RAM | arm64 |
| Docker | 29.8.2 + Compose v5.6.1 | Docker Desktop belum otomatis jalan |
| Python | 3.14.7 (system), 3.13.18 dipakai project | 3.13 dipilih karena wheel-nya lebih matang |
| AWS CLI | **tidak terpasang** | dan tidak ada `~/.aws` → belum ada kredensial |
| Git / gh | 2.54.0 / 2.97.0, login `SBryanK` | siap push |

**Keputusan penting:** karena tidak ada kredensial AWS, semua Bedrock call di proyek
ini **belum dijalankan**. Aku bangun dua mode: `local` (offline, buat demo/CI) dan
`bedrock` (siap dipakai begitu kamu `aws configure sso`). Jangan pernah mengarang
angka Bedrock — nanti kita ukur ulang setelah kredensial ada.

---

## 1. Docker basics (kira-kira 1 jam)

### Konsep 1 — image vs container

**Analogi:** image itu **resep + bahan yang sudah dibekukan** (blueprint, read-only).
Container itu **masakan yang disajikan** dari resep itu (instance yang jalan, punya
lapisan tulis sendiri). Satu image bisa menghasilkan banyak container.

```bash
docker images                       # daftar resep
docker ps                           # container yang sedang jalan
docker run --rm hello-world         # bikin container dari image, lalu hapus
```

Yang penting: container **ephemeral**. Kalau container dihapus, file yang ditulis di
dalamnya ikut hilang — kecuali disimpan di **volume** (nanti di Milestone 4).

### Konsep 2 — layers dan build cache

**Analogi:** image itu lasagna. Tiap instruksi Dockerfile = satu lapis. Lapis yang
tidak berubah dipakai ulang dari cache; kalau lapis bawah berubah, semua lapis di
atasnya harus dibuat ulang.

Itulah kenapa di Dockerfile produksi:

```dockerfile
COPY requirements.txt .        # lapis dependency — jarang berubah
RUN pip install -r requirements.txt
COPY app ./app                 # lapis kode — sering berubah
```

Hasil ukur di mesin kamu (production image):

| Perubahan | Waktu rebuild |
|---|---|
| tidak ada perubahan | **1.79 s** |
| ubah kode `app/` | **0.78 s** |
| ubah `requirements.txt` | **10.65 s** |

Kalau urutannya dibalik (`COPY app` dulu), ubah satu baris Python = install ulang
semua dependency. Itu bedanya 0.8 detik vs 10.6 detik.

### Konsep 3 — ports

**Analogi:** container itu rumah dengan pintu internal (port 8000 di dalam). `-p
8000:8000` artinya "pintu nomor 8000 di laptop disambungkan ke pintu 8000 di rumah".
Tanpa publish, container bisa saling lihat di network Docker, tapi laptop kamu tidak.

```bash
docker run --rm -p 8000:8000 policy-rag:latest
```

### Konsep 4 — environment variables

**Analogi:** env var itu **label/parameter** yang ditempel di container, bukan bagian
dari resep. Image-nya sama; perilakunya beda tergantung label (`LLM_PROVIDER=local`
vs `bedrock`). Di proyek ini semua config dibaca lewat `pydantic-settings`
(`app/config.py`), jadi tidak ada hard-code.

```bash
docker run --rm -e LLM_PROVIDER=local -p 8000:8000 policy-rag:latest
```

### Konsep 5 — volumes

**Analogi:** volume itu **lemari di luar rumah**. Rumah (container) boleh dibongkar,
isinya tetap aman. Di proyek ini: `qdrant_storage` menyimpan vector, `index_data`
menyimpan upload/index numpy.

### Praktik Milestone 1 — container FastAPI kecil

Jalankan image yang sudah jadi:

```bash
cd DOCKER-AWS-TRIAL
docker build -t policy-rag:latest .
docker run --rm -d --name policy-rag-demo -p 8000:8000 \
  -v "$PWD/data/policies:/data/policies:ro" policy-rag:latest
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/ingest/sample
curl -s -X POST http://localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question":"What is the NIST AI RMF Playbook?","top_k":4}'
docker rm -f policy-rag-demo
```

Yang harus kamu lihat: `/health` mengembalikan `chunks` (0 sebelum ingest), lalu
`/ingest/sample` mengubahnya jadi 719, dan `/ask` mengembalikan `answer` + `citations`.

### ✅ Check questions — Milestone 1 (English, jawaban Bahasa Indonesia)

1. **What is the difference between a Docker image and a container?**
   → Image = blueprint read-only (resep beku). Container = instance yang jalan dari
   image, punya writable layer sendiri. Satu image bisa jadi banyak container.
2. **Why does layer ordering matter for build cache, and what did we measure?**
   → Docker cache per layer; kalau layer berubah, layer di atasnya rebuild. Dependency
   ditaruh sebelum `COPY app`, jadi ubah kode cuma 0.78 s, sedangkan ubah dependency
   10.65 s (ukur beneran di mesin ini).
3. **What happens to data written inside a container when the container is removed?**
   → Hilang, karena container ephemeral. Solusinya volume (named volume/bind mount),
   seperti `qdrant_storage` yang menyimpan vector.

---

## 2. RAG app (kira-kira 3 jam)

### Konsep 1 — kenapa RAG, bukan fine-tuning?

**Analogi:** RAG itu **ujian open-book**. Model tidak perlu menghafal isi dokumen; kita
kasih potongan dokumen yang relevan tepat sebelum dia menjawab. Fine-tuning = menyuruh
model menghafal buku (mahal, cepat basi). RAG = kasih contekan yang bisa di-update
tanpa retrain.

### Konsep 2 — chunking

**Analogi:** buku 205 halaman itu terlalu besar buat "contekan" dan terlalu kasar buat
sitasi. Kita potong jadi kartu-kartu ringkas (±900 karakter, overlap 150 karakter),
seperti **kartu catatan**. Overlap penting supaya kalimat yang kepotong tetap punya
konteks.

Di proyek ini chunk **selalu mulai/berakhir di batas kalimat** (kecuali kalimat super
panjang), supaya snippet sitasi tidak mulai dari tengah kata. Itu bug yang sempat
muncul saat development dan sudah diperbaiki (`app/chunking.py`).

Hasil: 205 halaman → **719 chunk**.

### Konsep 3 — embedding

**Analogi:** embedding itu **koordinat makna**. "AI risk" dan "model danger" ditaruh
berdekatan di ruang 1024 dimensi; "banana" jauh. Cosine similarity mengukur sudut
antara dua vektor — makin kecil sudutnya, makin mirip.

Di mode Bedrock: `amazon.titan-embed-text-v2:0` (1024 dim) atau Cohere fallback. Di
mode offline: hashed bag-of-words deterministik (lexical, bukan semantik) — cukup buat
demo dan CI, tapi jelas lebih lemah. Itu sebabnya metrik kualitas offline harus
dibaca jujur.

### Konsep 4 — vector store

**Analogi:** embedding = koordinat; vector store = **indeks kartu katalog**. Kita simpan
vector + metadata (source, page, chunk_id). Pencarian = cari kartu dengan sudut paling
dekat.

- `numpy`: brute force dot-product, disimpan ke `data/index/vectors.npy` + `records.json`.
- `qdrant`: vector DB dalam container, REST API, named volume.

### Konsep 5 — grounding dan citations

**Analogi:** jawaban harus seperti **makalah dengan footnote**. Prompt melarang model
pakai pengetahuan luar; setiap klaim wajib `[n]`, dan API mengembalikan daftar sumber
dengan halaman. Kalau konteks tidak memuat jawaban, API bilang "tidak tercakup" —
bukan mengarang.

Prompt Bedrock ada di `app/llm.py`; citation parsing ada di `app/rag.py`. Field
`grounded` = ada minimal satu sitasi yang benar-benar dipakai.

### Konsep 6 — hybrid retrieval (bonus yang bikin beda)

**Analogi:** vector search itu "cari yang mirip maknanya"; BM25 itu "cari yang katanya
persis sama". Kalau digabung, kamu dapat yang mirip **dan** yang tepat. Di proyek ini:
ambil kandidat vector, lalu re-rank dengan BM25 (`app/retrieval.py`), lalu ambil top-k.
Di mode offline (hash lexical), bobot lexical dinaikkan karena vector-nya lemah.

### Kode yang perlu kamu baca

| File | Isi |
|---|---|
| `app/documents.py` | ekstraksi teks PDF per halaman (pypdf) |
| `app/chunking.py` | chunking sentence-aware + overlap + chunk_id stabil |
| `app/embeddings.py` | Bedrock Titan/Cohere + local hashed embedder |
| `app/store.py` | numpy store + Qdrant REST store |
| `app/retrieval.py` | BM25 rerank hybrid |
| `app/rag.py` | orkestrasi ingest & ask, citation parsing |
| `app/llm.py` | prompt + Converse API + extractive fallback |
| `app/main.py` | FastAPI routes, upload handling, health |

### Praktik Milestone 2

```bash
.venv/bin/python -m scripts.ingest --reset          # 719 chunk, ~1.7 s
.venv/bin/uvicorn app.main:app --reload
# buka http://localhost:8000
```

Coba tanya: *"What are the four core functions of the NIST AI Risk Management
Framework?"* lalu lihat citation card di UI (source, page, score, chunk_id).

### ✅ Check questions — Milestone 2

1. **Why chunk documents instead of embedding whole PDFs?**
   → Batas context window, presisi retrieval, sitasi per halaman, dan biaya embedding
   lebih terkendali. Trade-off: ukuran chunk & overlap harus dievaluasi.
2. **What does an embedding represent, and why is cosine similarity used?**
   → Representasi vektor dari makna/lexical content; cosine mengukur arah (sudut),
   bukan panjang, sehingga stabil terhadap panjang teks.
3. **How does the API stay grounded, and how could it still fail?**
   → Prompt hanya boleh pakai konteks, wajib sitasi, ada jalur "not covered", dan flag
   `grounded`. Bisa tetap gagal kalau retrieval salah atau model mengabaikan instruksi
   — karena itu ada evaluasi dan (di production) groundedness check.

---

## 3. Good Dockerfile (kira-kira 1 jam)

### Konsep 1 — multi-stage build

**Analogi:** **dapur** (builder) vs **meja makan** (runtime). Di dapur ada kompor,
panci, dan sisa bahan (build tools, cache pip). Yang dibawa ke meja makan cuma
masakan jadi (venv + kode). Multi-stage = buang dapur dari image akhir.

```dockerfile
FROM python:3.13-slim AS builder
RUN python -m venv /opt/venv && pip install -r requirements.txt

FROM python:3.13-slim AS runtime
COPY --from=builder /opt/venv /opt/venv
COPY app ./app
```

### Konsep 2 — slim base image

`python:3.13` penuh membawa compiler, headers, test tools; `python:3.13-slim` cukup
untuk runtime. Akibatnya: **1.83 GB → 429 MB (−77%)**. Ini bukan teori; hasil
`docker image inspect`.

### Konsep 3 — non-root user

**Analogi:** tamu vs pemilik rumah. Kalau aplikasi di-hack, penyerang cuma jadi "tamu"
(uid 10001), bukan root yang bisa mengubah seluruh container/host mount.

```dockerfile
RUN groupadd --gid 10001 app && useradd --uid 10001 --gid app app
USER app
```

Verifikasi: `docker inspect policy-rag-api --format '{{.Config.User}}'` → `app`.

### Konsep 4 — HEALTHCHECK

**Analogi:** lampu indikator di dashboard. Docker bisa tahu aplikasi benar-benar siap,
bukan cuma "proses masih hidup". `--start-period=20s` memberi waktu boot sebelum
kegagalan dihitung; `--retries=3` menghindari false alarm.

```dockerfile
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
```

Compose memakai `condition: service_healthy` supaya API tidak start sebelum Qdrant siap.

### Konsep 5 — .dockerignore

Build context dikirim ke daemon; kalau `.git`, `.venv`, `data`, `docs` ikut, build lambat
dan rawan membocorkan `.env`. `.dockerignore` memangkas context dan mencegah secrets
masuk image.

### Praktik Milestone 3

```bash
docker build -f docker/Dockerfile.naive -t policy-rag:naive .
docker build -t policy-rag:latest .
docker images --format '{{.Repository}}:{{.Tag}} {{.Size}}' | grep policy-rag
```

### ✅ Check questions — Milestone 3

1. **Why does a multi-stage build reduce image size?**
   → Build tools, compiler, dan cache hanya ada di stage builder; stage runtime hanya
   menerima venv + kode. Terukur 1.83 GB → 429 MB.
2. **Why run as a non-root user, and how do you verify it?**
   → Mengurangi blast radius kalau app compromised. Verifikasi via
   `docker inspect ... {{.Config.User}}` atau `docker exec ... id`.
3. **What does HEALTHCHECK add over "container is running"?**
   → Status kesiapan aplikasi (HTTP 200), dipakai orchestrator untuk restart dan untuk
   `depends_on: condition: service_healthy`.

---

## 4. docker compose (kira-kira 1 jam)

### Konsep 1 — compose sebagai conductor

**Analogi:** compose itu **konduktor orkestra**: satu file mendefinisikan service
(api, qdrant), network, dan volume, lalu menyalakan semuanya dengan satu perintah.
Container saling menemukan lewat **DNS internal** — `http://qdrant:6333`, bukan
`localhost`.

### Konsep 2 — named volume & persistence

**Analogi:** lemari arsip di luar gedung. `docker compose down` membongkar gedung
(container) tapi lemari (`qdrant_storage`) tetap. `docker compose down -v` baru
menghapus lemari.

Bukti yang kita ukur:

| Aksi | Hasil |
|---|---|
| ingest ke Qdrant | 719 points |
| `docker compose restart` | 719 chunks, 719 points |
| `docker compose down && docker compose up -d` | 719 chunks, 719 points |
| `/ask` setelah up | tetap mengembalikan sitasi |

### Konsep 3 — healthcheck dependency

`depends_on` biasa hanya urutan start; `condition: service_healthy` menunggu Qdrant
benar-benar siap menerima request. Tanpa ini, API bisa crash saat startup karena
Qdrant belum listen.

### Praktik Milestone 4

```bash
docker compose up -d --build
docker compose ps
curl -X POST http://localhost:8000/ingest/sample
curl -s http://localhost:8000/stats
docker compose restart
curl -s http://localhost:8000/health
docker compose down && docker compose up -d
curl -s http://localhost:8000/health
```

### ✅ Check questions — Milestone 4

1. **How do containers in the same compose network find each other?**
   → Lewat DNS service name di network default compose: `http://qdrant:6333`.
2. **What is a named volume, and how did we prove persistence?**
   → Storage yang lifecycle-nya terpisah dari container; setelah restart dan
   `down`/`up`, tetap 719 points. `down -v` yang menghapus.
3. **Why `condition: service_healthy` instead of plain `depends_on`?**
   → Menjamin readiness, bukan sekadar urutan start; mencegah API start sebelum Qdrant
   siap.

---

## 5. Ship (kira-kira 2 jam)

### Konsep 1 — README & website portfolio

README harus menjawab: masalah apa, arsitektur, cara jalanin, hasil ukur, biaya,
keamanan, trade-off, dan apa yang berubah di skala enterprise. Website GitHub Pages
(`docs/index.html`) adalah versi visual untuk reviewer yang tidak mau baca repo.

### Konsep 2 — evaluasi yang jujur

10 pertanyaan, masing-masing punya `expected_source` dan `expected_terms`. Yang diukur:

- **latency** (client-side): p50 4,8 ms, p95 7,6 ms (offline mode, stack Compose/Qdrant);
- **source retrieved**: 100% (top-4);
- **source cited**: 100%;
- **term coverage**: 58,3%.

Kenapa term coverage cuma 58%? Karena mode offline **menyalin kalimat**, bukan menjawab
fluent. Itu batas atas mode lokal, bukan angka Bedrock. Angka Bedrock wajib diukur ulang
setelah kredensial ada. Jangan pakai angka 58% di resume sebagai "answer quality Bedrock".

### Konsep 3 — biaya & model lifecycle

Biaya Bedrock = token × harga per 1M. Dari AWS Price List ap-southeast-1: Nova Lite
$0.081/M input, $0.324/M output; Titan embedding $0.028/M. Satu full run (ingest 120k
token + 10 pertanyaan) ≈ **$0.005**. Tetap pasang budget alert USD 5 sebelum call
pertama.

Pelajaran model lifecycle: `anthropic.claude-3-haiku-20240307-v1:0` **EOL di
ap-southeast-1 pada 2026-11-13** — tutorial lama akan gagal. Karena itu repo ini
punya `scripts/check_bedrock.py` untuk verifikasi ID/ketersediaan dari akun kamu.

### Konsep 4 — enterprise scale

Yang berubah: vector DB terkelola, ingestion async dari S3, hybrid + reranker,
auth/rate limit, observability (trace + token cost), IaC, multi-region, data residency,
guardrails, dan eval retrieval di CI.

### ✅ Check questions — Milestone 5

1. **What does the evaluation actually measure, and what is its main limitation?**
   → Retrieval hit, citation hit, latency, dan lexical term coverage. Batasnya: term
   coverage bukan penilaian kualitas manusia, dan angka saat ini dari mode extractive
   offline, bukan Bedrock.
2. **Why verify Bedrock model IDs and availability instead of hard-coding them?**
   → Region availability, inference profile, EOL, dan quota berubah; contoh nyata
   Claude 3 Haiku EOL dua minggu sebelum build. Verifikasi lewat akun sendiri.
3. **Name one trade-off you would change first at enterprise scale, and why.**
   → Misal ganti numpy brute-force + BM25 ke managed vector DB + reranker, karena
   latency/recall tidak scalable dan butuh HA; atau tambah auth/rate limit karena API
   publik tanpa proteksi.

---

## 5.1 Website portfolio: tema terang/gelap + Bahasa Indonesia

Website ada di `docs/index.html` dan otomatis jadi GitHub Pages. Fitur barunya:

- **Tombol tema** (ikon ☀️/🌙) di kanan atas: ganti dark ↔ light, pilihan disimpan di
  `localStorage` browser.
- **Tombol bahasa** (`ID`/`EN`): ganti seluruh halaman ke Bahasa Indonesia atau Inggris,
  termasuk judul tab browser dan meta description.
- **URL pintasan** untuk share/QA: `?theme=light&lang=id` atau `?theme=dark&lang=en`.
  Contoh: `https://sbryank.github.io/DOCKER-AWS-TRIAL/?theme=light&lang=id`.
- Default: mengikuti `prefers-color-scheme` laptop dan bahasa browser.

**Cara kerjanya (analogi anak 10 tahun):** halaman punya "kamus" dua bahasa
(`const I18N = { en: {...}, id: {...} }`). Setiap teks diberi label `data-i18n="kunci"`.
Saat tombol bahasa diklik, JavaScript mengambil kalimat dari kamus yang cocok dan
mengganti isi labelnya. Tema bekerja mirip: satu saklar di `<html data-theme="light|dark">`
mengubah daftar warna (`CSS variables`) untuk seluruh halaman.

**Deploy sendiri (langkah persis):**

1. Edit `docs/index.html` (tema/warna/teks) di laptop.
2. `git add -A && git commit -m "update website" && git push`.
3. Buka repo GitHub → **Settings → Pages**.
4. Bagian **Build and deployment**: Source = **Deploy from a branch**,
   Branch = **main**, folder = **/docs**, lalu **Save**.
5. Tunggu tab **Actions** → workflow *pages build and deployment* selesai (±30 detik).
6. Buka `https://sbryank.github.io/DOCKER-AWS-TRIAL/` dan coba kedua tombol di kanan atas.

**Menambah bahasa baru (misal Jepang):** tambahkan blok `ja: { ... }` di objek `I18N`
dengan kunci yang sama, lalu tambahkan tombol/siklus bahasa di JavaScript. Tidak perlu
framework, tidak perlu build step.

**Test otomatis website:** screenshot dan pengecekan tema/bahasa dilakukan dengan headless
Chrome (DevTools Protocol): klik tombol, cek atribut `data-theme`/`lang`, cek `localStorage`,
dan memastikan **0 exception JavaScript**. Hasilnya `passed: true`.

## 5.2 Functional test: 29/29 lolos

Jalankan (stack harus hidup):

```bash
docker compose up -d --build
.venv/bin/python -m scripts.functional_test
```

Yang diuji:

- `/`, `/docs`, `/openapi.json`, `/health`, `/stats`;
- `/ingest/sample` (3 dokumen, 205 halaman, 719 chunk, <10 detik);
- `/ask` dengan sitasi + sumber NIST + latensi;
- pertanyaan di luar korpus → menolak dengan jujur, bukan mengarang;
- validasi 422 (pertanyaan pendek, `top_k` 99, field kosong);
- upload `.exe` dan file kosong → 400;
- upload `.txt` + reset → hanya dokumen itu yang tersimpan, lalu bisa ditanya;
- restore korpus sampel → kembali 719 chunk;
- 404 untuk route tidak dikenal.

**Yang belum bisa dites tanpa akun AWS** (nanti setelah `aws configure sso`):
`scripts/check_bedrock.py` (model access, ID, Converse, budget), `scripts/setup_budget.py`
(pembuatan budget USD 5), dan kualitas jawaban Bedrock asli lewat `scripts/run_eval.py`.
Tanpa kredensial, ketiga script itu memang menampilkan pesan error yang jelas — itu
perilaku yang benar, bukan bug.

---

## 6. Troubleshooting

| Gejala | Penyebab | Solusi |
|---|---|---|
| `failed to connect to docker API` | Docker Desktop belum jalan | buka Docker Desktop, tunggu `docker info` sukses |
| `port is already allocated` | ada server lain di 8000 | `lsof -i :8000`, matikan prosesnya |
| `No extractable text` | PDF hasil scan | OCR dulu (contoh `ocrmypdf`) |
| `Cannot reach Qdrant` | Qdrant belum sehat | `docker compose ps`, cek healthcheck |
| `AccessDeniedException` | IAM/model access | cek `bedrock:InvokeModel`, region, `check_bedrock` |
| `ResourceNotFoundException` | model tidak ada/EOL | ganti ID sesuai hasil `check_bedrock` |
| Jawaban "not covered" padahal ada | retrieval meleset | naikkan `TOP_K`, cek chunk, evaluasi ulang |
| Image besar | pakai image full / tidak multi-stage | bandingkan dengan `docker/Dockerfile.naive` |

---

## 7. Resume kerja (kalau buka chat baru)

1. `cd DOCKER-AWS-TRIAL && docker compose up -d --build`
2. `curl -X POST localhost:8000/ingest/sample`
3. Baca `PROGRESS.md` untuk status terakhir dan open issues.
4. Untuk Bedrock: `aws configure sso` → `python -m scripts.setup_budget --email ...` →
   `python -m scripts.check_bedrock --profile ...` → set env provider ke `bedrock`.
5. Re-run eval dan update README dengan angka Bedrock yang asli.

---

## 8. Glossary singkat

- **Image / container / layer / volume** — lihat Milestone 1.
- **RAG** — retrieval-augmented generation; model menjawab dari konteks yang diambil.
- **Chunk** — potongan dokumen yang di-embed dan di-retrieve.
- **Embedding** — vektor representasi teks.
- **Cosine similarity** — kemiripan arah dua vektor.
- **Top-k** — jumlah chunk yang diambil sebagai konteks.
- **Grounding / citation** — jawaban harus berasal dari konteks dan menyebut sumber.
- **Hybrid retrieval** — gabungan vector search + lexical (BM25).
- **Inference profile** — ID Bedrock yang merutekan request lintas region (mis. `apac.`).
- **Multi-stage build** — beberapa `FROM` dalam satu Dockerfile; hanya stage akhir yang dikirim.
