# Dockerfile - headless OpenCV preinstalled
FROM python:3.10-slim

WORKDIR /app

# Upgrade pip and install headless OpenCV first
RUN pip install --upgrade pip
RUN pip install opencv-python-headless==4.10.0.84

# Copy requirements and install the rest of Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy app code
COPY . /app

# Set Streamlit server env to avoid inotify issues
ENV STREAMLIT_SERVER_FILEWATCHERTYPE=none

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.fileWatcherType", "none"]

