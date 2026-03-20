FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code, data, and tests
COPY automation.py config.py stages.py utils.py outputs.py ./
COPY test_automation.py ./
COPY tenant_inquiries.csv lease_clauses.json ./

# Create output directory
RUN mkdir -p sample_io

# Default command: run the full pipeline
# To run tests: docker run --rm automation python3 -m pytest test_automation.py -v
ENTRYPOINT ["python3", "automation.py"]
