# llama-rag


Run ```./create-venv.sh``` to create the python 3.10 environment.


To create the Milvus vector store, run the ```./milvus.setup.sh```

Activate ```venv``` using ```source venv/bin/activate```.
Download the SLM model and Embedding model first using ```python get-slm.py```.
Run ```python milvus_test.py``` to verify Milvus connection and document indexing.
Run ```python test-slm.py``` to test the Milvus retrieval as well as SLM model inference.