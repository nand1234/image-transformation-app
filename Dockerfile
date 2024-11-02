# Stage 1: Build stage
FROM python:3.10 AS build-stage

# Ensure pip3 is up to date (install pip3 if needed)
RUN apt-get update && apt-get install -y python3-pip && pip3 install --upgrade pip

# Set the working directory inside the container to /app
WORKDIR /app

# Copy requirements.txt into the container
COPY requirements.txt .

# Install dependencies from requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the source code and checkpoint files to the build stage
COPY checkquality.py app.py detect_watermark.py run_real_esrgan.py /app/
COPY templates /app/templates/

# Copy weights folder
COPY weights /app/weights

# Stage 2: Final runtime stage
FROM python:3.10-slim AS runtime-stage

# Set the working directory inside the container to /app
WORKDIR /app

# Copy installed Python dependencies from build stage to runtime stage
COPY --from=build-stage /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=build-stage /usr/local/bin /usr/local/bin

# Copy only the necessary files for running the app
COPY --from=build-stage /app /app

# Expose the port the app runs on
EXPOSE 5000

# Command to run your application
CMD ["python3", "app.py"]
