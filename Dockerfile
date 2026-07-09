FROM python:3.12-slim
WORKDIR /app
COPY server.py .
RUN pip install --no-cache-dir "mcp>=1.2.0"
ENV MCP_HTTP=1 HOST=0.0.0.0 PORT=8000
EXPOSE 8000
# Remote MCP endpoint served at /mcp (streamable-http)
CMD ["python", "server.py", "--http"]
