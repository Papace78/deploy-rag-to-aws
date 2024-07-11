FROM public.ecr.aws/lambda/python:3.11

COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Pixegami hack.
# It is requireed to make SQLite3 work for Chroma.
RUN pip install pysqlite3-binary


RUN pip install -r requirements.txt --upgrade

# For local testing.
EXPOSE 8000


# Pixegami hack.
# Set IS_USING_IMAGE_RUNTIME Environment Variable
ENV IS_USING_IMAGE_RUNTIME=True

COPY src/* ${LAMBDA_TASK_ROOT}
COPY src/rag_app ${LAMBDA_TASK_ROOT}/rag_app
COPY src/data/chroma ${LAMBDA_TASK_ROOT}/data/chroma
