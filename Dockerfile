FROM python:3.12-slim

# System-Abhängigkeiten: ca-certs + Chromium-Runtime-Libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libdbus-1-3 \
    libxcb1 \
    libxkbcommon0 \
    libx11-6 \
    libxcomposite1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python-Abhängigkeiten + Chromium (ohne --with-deps, da Libs oben schon installiert)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    playwright install chromium

# Anwendungscode kopieren
COPY main.py .
COPY dashboard.py .
COPY src/ ./src/
COPY config/ ./config/

# data/ wird NICHT kopiert – kommt als Bind Mount rein (CV, Logs, Analysen)

CMD ["python", "main.py", "--help"]
