FROM python:3.10-slim

WORKDIR /app

# small system dependencies for pip wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# upgrade pip and install headless OpenCV first (force headless only)
RUN pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir opencv-python-headless==4.10.0.84

# copy requirements (must NOT include opencv)
COPY requirements.txt .

# install rest with retries/timeouts to avoid broken pipe
RUN pip install --no-cache-dir --retries 8 --timeout 120 -r requirements.txt

# copy code and start
COPY . /app
ENV STREAMLIT_SERVER_FILEWATCHERTYPE=none
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.fileWatcherType", "none"]
