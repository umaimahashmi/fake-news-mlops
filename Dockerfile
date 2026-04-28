FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY ml/ ./ml/

RUN mkdir -p models data

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--timeout-keep-alive", "30"]
