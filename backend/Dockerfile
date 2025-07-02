FROM python:3.11-slim

WORKDIR /backend

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ /backend

ENV FLASK_APP=app.py
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
