from langchain_aws import BedrockEmbeddings


def get_embedding_function():
    embeddings = BedrockEmbeddings(model_id = "amazon.titan-embed-text-v1", region_name = "us-east-1")
    return embeddings
