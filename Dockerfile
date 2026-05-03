FROM python:3.11-slim

WORKDIR /app

RUN pip install uv --no-cache-dir

COPY pyproject.toml uv.lock ./
RUN uv sync --dev --no-install-project --no-cache

COPY src/ ./src/
COPY langgraph.json ./

RUN uv pip install -e . --no-cache

# langgraph.json 引用 ./.env，容器里提供空文件避免报错
# 真实环境变量通过 docker run --env-file 传入
RUN touch .env

RUN mkdir -p /data

EXPOSE 8123

CMD ["uv", "run", "langgraph", "dev", "--host", "0.0.0.0", "--port", "8123", "--no-browser"]
