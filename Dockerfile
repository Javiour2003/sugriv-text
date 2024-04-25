# Stage 1: Build environment using a Python base image
FROM python:3.12 as builder

# Install build tools
RUN apt-get update && apt-get install -y gcc g++ cmake zip

# Copy requirements.txt and install packages with appropriate CMAKE_ARGS
COPY requirements.txt .
RUN CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" pip install --upgrade pip && pip install -r requirements.txt

# Stage 2: Final image using AWS Lambda Python image
FROM public.ecr.aws/lambda/python:3.12

# copy all code
COPY . .

# Copy installed packages from builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages/ /var/lang/lib/python3.12/site-packages/

# Copy lambda function code
COPY main.py ${LAMBDA_TASK_ROOT}

EXPOSE 8080

CMD [ "main.handler" ]
