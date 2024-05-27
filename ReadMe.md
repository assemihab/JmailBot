# JmailWriter: Tailoring Prompts to Elegant Writing

**Introduction:**

in this project, we aim to guide the user in writing prompts that invoke the LLM to:

- Generate emails
- Modify emails
- Adjust meeting times
- Write cover letters
- Retrieve information about companies

![image](images/task.jpg)

##Prompt engineering Tools utilized:

- RAG
- Output Parsing
- Memory
- Few-shot-learning
- Query Routing

## Requirements:

- A working OpenAI key in this line ` os.environ['OPENAI_API_KEY']='KEY' `
- All necessary libraries listed in requirements.txt

**Example on Using RAG and memory**

![image](images/Rag.jpg)

**Example on Generating Tailored cover letter and parsed output **
![image](images/coverletter.jpg)


![image](images/outputparsing.jpg)


Before executing the project, ensure that you have pip installed in your environment. You can verify this by running the following command in the command prompt:

	pip --version

Then, install dependencies and run the project with the following commands:

```
pip install -r requirements.txt
py GUI.py
```