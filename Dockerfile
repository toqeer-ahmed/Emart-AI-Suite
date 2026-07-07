FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Persistent-ish demo storage (mount a volume here in docker-compose so the
# SQLite demo DB and generated photos survive container restarts)
RUN mkdir -p /app/app/shared /app/app/services/photo_studio/output

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
