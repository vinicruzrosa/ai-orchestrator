FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY proto/ proto/
RUN mkdir -p app/adapters/proto_generated && \
    python -m grpc_tools.protoc -I=proto --python_out=app/adapters/proto_generated proto/analysis.proto && \
    touch app/adapters/proto_generated/__init__.py

COPY . .

CMD ["python", "main.py"]
