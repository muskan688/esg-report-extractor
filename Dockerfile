FROM python:3.11-slim

WORKDIR /app

# libgl/libglib are pulled in by some pdfplumber/PIL transitive deps on slim images
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY src/ src/
COPY app/ app/
COPY scripts/ scripts/
RUN pip install --no-cache-dir -e .

ENV ESG_DATA_DIR=/app/data
VOLUME ["/app/data"]

EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
