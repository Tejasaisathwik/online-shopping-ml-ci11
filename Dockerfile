FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train_model.py .
COPY quality_gate.py .
COPY app.py .
COPY test_ml_pipeline.py .
COPY test_app.py .
COPY data/ data/

RUN python train_model.py
RUN python quality_gate.py

EXPOSE 5000

CMD ["python", "app.py"]