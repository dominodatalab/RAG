# RAG

## Files
*  RAG.ipynb : This notebook contains all the logic to load embeddings, setup the QA chain for RAG and allows users to ask queries once all the hyperparameters have been finalized.
  
*  MLflow_eval.ipynb : This notebook contains code to evaluate a RAG pipeline for faithfulness and relevance using Mlflow. The metrics are also stored and can be visualized in the Experiments tab in Domino
  
*  RAGAS_eval.ipynb : This notebook uses the RAGAS package to evaluate a RAG pipeline. This is another example of how to evaluate a RAG pipeline, RAGAS offers a couple of more metrics than MLFlow
  
*  example_prompts.txt : Has a few examples of questions that can be presented as prompts to the QA chain
  
*  app.sh : Script required to setup and use Streamlit in Domino
  
*  streamlit_app.py : This file contains code that sets up the UI and workflow for a Streamlit chatbot. The app needs an Anthropic and Qdrant key to set in the sidebar to run

On `se-demo` this was run on a `Medium` hardware tier

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
On `se2-demo` this environment is available as `MedRAG`

### 

