# LocalRAG

## Environment Setup

### Custom base image 
```Domino Standard Environment Py3.9 R4.2```


### Dockerfile instructions

```
USER root:root

RUN pip uninstall --yes mlflow

RUN pip install openai langchain  transformers tiktoken  sentence-transformers \
                qdrant-client ragas mlflow==2.8.0 getpass4 anthropic evaluate \
                textstat streamlit pypdf accelerate peft bitsandbytes

RUN pip install -i https://test.pypi.org/simple/ streamlit-chat-domino
```
### 

