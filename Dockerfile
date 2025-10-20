
# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Install build essentials including cmake
RUN apt-get update && apt-get install -y build-essential cmake

# Install uv
RUN pip install uv

# Copy the pyproject.toml and uv.lock files into the container at /app
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
RUN uv sync

# Copy the rest of the application's code into the container at /app
COPY . .

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define environment variable
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Run app.py when the container launches
CMD ["/app/.venv/bin/streamlit", "run", "app.py"]
