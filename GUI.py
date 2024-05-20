# %% [markdown]
# # environment setup

# %%
# !pip install langchain
# !pip install langchain-openai
# !pip install openai
# ! pip install langchain-chroma


path='E:/FCSE/3. big data lab/lab project/lol.json'
pathtocompany = "E:/FCSE/3. big data lab/lab project/companies.csv"
db_dir="E:/FCSE/3. big data lab/lab project/Chroma_DB"
# pathtocompany = "/content/companies.csv"
# db_dir="/content/Chroma_DB"

# %% [markdown]
# # imports and global codes

# %%
from langchain_openai import OpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field

from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.memory import ConversationEntityMemory
from langchain.memory import ConversationBufferWindowMemory,ChatMessageHistory
from langchain.chains import ConversationChain
from langchain.schema import messages_from_dict, messages_to_dict
import json
import pandas as pd
from typing import Literal 
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.pydantic_v1 import BaseModel, Field 
from langchain_openai import ChatOpenAI 
from langchain_community.document_loaders import DataFrameLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# %%
llm=OpenAI()

# %% [markdown]
# # output parser

# %%
class Email(BaseModel):
    from_email: str = Field(description="The email address of the sender")
    to_email: str = Field(description="The email address of the recipient")
    subject: str = Field(description="The subject of the email")
    body: str = Field(description="The body of the email")
parser=PydanticOutputParser(pydantic_object=Email)

# %% [markdown]
# # templates

# %%
# add template for cover letters



correctingTemp=PromptTemplate(template="enhance this email with this subject:'{subject}' body:\n'{body}' to make it more professional don't change the meaning of the email or length too much"
                              ,input_variables=['subject','body']
                              )
#specify the length of the email and email of sender add the mail tag
writingFromScratchTemp=PromptTemplate(template="write a professional email with this subject:{subject} to {Temail} from {femail} with sender:{sender} reciver:{reciver}\n{format_instructions}"
                                      ,input_variables=['subject','Temail','femail','sender','reciver']
                                      ,partial_variables={'format_instructions':parser.get_format_instructions()})
coverLetterTemp=PromptTemplate(template="write a cover letter for the job of {job} at {company} with sender: {name} and from email {fmail} to {Tmail} \n{format_instructions}"
                                ,input_variables=['job','company','name','fmail','Tmail']
                                ,partial_variables={'format_instructions':parser.get_format_instructions()})
changeorsetupmeetingTemp=PromptTemplate(template="write an email to {Temail} from {femail} with subject:{subject} to setup or change a meeting time at {date}\n{format_instructions}"
                                        ,input_variables=['Temail','femail','subject','date']
                                        ,partial_variables={'format_instructions':parser.get_format_instructions()})
example_prompt = PromptTemplate(input_variables=["example"],
template="example Written by me: {example}")



# %% [markdown]
# # fewshot learning
# 

# %%
coverLetterTemp2=PromptTemplate(template="write a cover letter for the job of {job} at {company} with sender: {name} and from email {fmail} to {Tmail} email: \n{format_instructions}"
                                ,input_variables=['job','company','name','fmail','Tmail']
                                ,partial_variables={'format_instructions':parser.get_format_instructions()})

# %% [markdown]
# # memory

# %%
secondClettertemp=PromptTemplate(template="base on previous mails write a cover letter for the job of {job} at {company} with reciver:{reciver} and sender:{sender}  "
                                ,input_variables=['job','company','reciver','sender'])
modifyMailTemp=PromptTemplate(template='modify and resend this the last email where the part to be modified is: "{part}" to be {new_part}',
                              input_variables=['part','new_part'],
                              format_instructions=parser.get_format_instructions()
                              )
coverConversation=ConversationChain(
llm=llm,
memory=ConversationBufferWindowMemory(k=4),
verbose=False
)

# %% [markdown]
# # using rag

# %%


# # # pathtocompany='E:/FCSE/3. big data lab/lab project/companies.csv'

# companiesdf = pd.read_csv(pathtocompany)
# exclude_columns = ['Company', 'Company Name for Emails']
# companies_info = [col for col in companiesdf.columns if col not in exclude_columns]

# companiesdf.head()

# %%
# loader = DataFrameLoader(companiesdf, page_content_column="Company")
# companies_data=loader.load()

# %%
# companies_data[0]

# %%
# list( companies_data[0] )

# %%
# # 

# embeddings_model = OpenAIEmbeddings()

#  #Get the data of the all chunks and emedding them
# docs_chunks = [chunk.page_content for chunk in companies_data ]
# docs_embeddings = embeddings_model.embed_documents(docs_chunks)

# %%
# len(companies_data)

# %%
# 

# db_dir="/content/Chroma_DB"

#generate Ids for each chunck note:IDs must be string
# chunks_IDs= [str(id) for id in (list(range(len(companies_data))))]
# chroma_db= Chroma.from_documents(companies_data,
#                                  embeddings_model,
#                                  ids=chunks_IDs,
#                                  persist_directory=db_dir)

# %%
embeddings_model = OpenAIEmbeddings()
loaded_vector_db = Chroma(
    persist_directory=db_dir,
    embedding_function=embeddings_model
)

# %%
user_input = "I recently had a great experience with Hassan Allam Holding company. I wonder what does their Annual Revenue?."
similar_docs = loaded_vector_db.similarity_search(user_input)
similar_docs[0].metadata

# %%
def extract_company_info(metadata ,userinput):
    if len(coverConversation.memory.chat_memory.messages)==0:
        
      prompt_template = PromptTemplate.from_template(
        "Extract only the wanted info from the following dictionary:{anydata} that match the user needs ({needs}) "
      )
      string_prompt_message=prompt_template.format(anydata=metadata ,needs=userinput )
    elif len(coverConversation.memory.chat_memory.messages)>4:
      coverConversation.memory.clear()
      prompt_template = PromptTemplate.from_template(
        "Extract only the wanted info from the following dictionary:{anydata} that match the user needs ({needs}) "
      )
      string_prompt_message=prompt_template.format(anydata=metadata ,needs=userinput )
    else:
      prompt_template = PromptTemplate.from_template(
        "Extract only the wanted info from the previous dictionary that match the user needs ({needs}) "
      )
      string_prompt_message=prompt_template.format( needs=userinput )
    company_info = coverConversation.predict(input=string_prompt_message)

    return company_info
def output_info(input):
  similar_doc = loaded_vector_db.similarity_search(input)[0].metadata
  return extract_company_info(similar_doc ,input)

# user_input = "I recently had a great experience with Hassan Allam Holding company. I wonder what is their Annual Revenue?."
# output_info(user_input)


# %%
user_input2 = "What are the number of Employees of WUZZUF company"
output_info(user_input2)

# %%


# %%
coverConversation.memory.clear()

# %%
coverConversation.memory.chat_memory.messages

# %%
len(coverConversation.memory.chat_memory.messages)

# %%
# messages=coverConversation.memory.chat_memory.messages

# %%
# user_input2 = "What are the number of workers of WUZZUF company"
# output_info(user_input2)

# %% [markdown]
# 

# %% [markdown]
# Rag from user's insertedFile

# %% [markdown]
# query routing

# %%
class RouteQuery(BaseModel): 
    """Route a user query to the most relevant datasource.""" 
    templatetouse: Literal['chooseAtask',"generateEmail", "modifyEmail", "generatecoverletter","getcompanyinformation",'changetime'] = \
    Field( ..., description="Given a user request choose which template would be most relevant for satisfying their request", ) 
llm2 = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
structured_llm = llm2.with_structured_output(RouteQuery)
system1="you are a job helper system, I can help you with generating emails, modifying emails, generating cover letters, getting company information, and changing meeting times"
chatrouting = ChatPromptTemplate.from_messages( 
[ 
("system", system1), 
("human", "{request}"), 
] 
) 
router = chatrouting | structured_llm 


# %%
class routeQuery2(BaseModel):
    fewshotbool: bool = Field( ..., description="Whether to use fewshot or not", )
structured_llm2=llm2.with_structured_output(routeQuery2)
system2="your job is to determine whether to use fewshot or not"
fewshotrouting = ChatPromptTemplate.from_messages(
    [
        ("system", system2),
        ("human", "{request}"),
    ]
)
router2 = fewshotrouting | structured_llm2

# %%
# testquery="yes use fewshot" 


# %% [markdown]
# testing query routing

# %%


# %%
request1 = "send email"
response=router.invoke({"request": request1})
print(type(response.templatetouse))
print(response.templatetouse)
request2="yes use it"
response2=router2.invoke({"request":request2})
print(response2)
print(type(response2.fewshotbool))

# %%
from tkinter import *
import customtkinter as cm
from tkinter import scrolledtext
import threading

cm.set_appearance_mode("dark")
cm.set_default_color_theme("dark-blue")

root = Tk()
root.geometry("500x600")
root.title("LLM Project")
messagetype=0
generatingMailSteps=modifyMailstep=coverletterstep=changetimestep=0

subject=tmail=fmail=sender=receiver=None
modsubject=Modbody=None
job=company=covname=covfmail=covtmail=covreciver=None
timetmail=timefmail=timesubject=timedate=None
fewshotExamples=0
examples=[]
def modifyEmailSend(user_message):
    global modsubject,modbody,messagetype,modifyMailstep
    if modifyMailstep==1:
        modsubject=user_message
        return None
    elif modifyMailstep==2:
        modbody=user_message
        modifyEmail=modifyMailTemp.format(part=modsubject,new_part=modbody)
        modifyEmailResponse=llm.invoke(modifyEmail)
        display_message("Bot", modifyEmailResponse)
        messagetype=0
        modifyMailstep=0
        return 'return to main menu'
def modifyMailrespond():
    global modifyMailstep
    if modifyMailstep==0:
        modifyMailstep=1
        return "what is subject of the email you want to modify"
    elif modifyMailstep==1:
        modifyMailstep=2
        return "what is the body of email you want to modify"
def coverletterRespond():
    global coverletterstep,fewshotExamples
    if coverletterstep==0:
        coverletterstep=1
        return "what is the job you are applying for"
    elif coverletterstep==1:
        coverletterstep=2
        return "what is the company you are applying for"
    elif coverletterstep==2:
        coverletterstep=3
        return "what is your name"
    elif coverletterstep==3:
        coverletterstep=4
        return "what is your email"
    elif coverletterstep==4:
        coverletterstep=5
        return "what is the recipient email"
    elif coverletterstep==5:
        coverletterstep=6
        return "how many fewshot examples do you want from zero to hero"
    elif coverletterstep==6 or coverletterstep==7:
        if fewshotExamples>0:
            coverletterstep=7
            fewshotExamples=fewshotExamples-1
            return "please provide another example"
        else:
            coverletterstep=8
            return "write anything to generate the cover letter"
def coverletterSender(user_message):
    global job,company,covname,covfmail,covtmail,covreciver,messagetype,coverletterstep,fewshotExamples
    if coverletterstep==1:
        job=user_message
    elif coverletterstep==2:
        company=user_message
    elif coverletterstep==3:
        covname=user_message
    elif coverletterstep==4:
        covfmail=user_message
    elif coverletterstep==5:
        covtmail=user_message
    elif coverletterstep==6:
        fewshotExamples=int(user_message)
    elif coverletterstep==7:
        example=user_message
        examples.append({"example":example})
    elif coverletterstep==8:
        covershotprompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="write a cover letter for the job of {job} at company: {company} with reciver:{reciver} and sender:{sender} and  and from email {fmail} to {Tmail} \n{format_instructions}",
    input_variables=["input"],
    partial_variables={'format_instructions':parser.get_format_instructions()}
)
        covermessage=covershotprompt.format(job=job,company=company,sender=covname,fmail=covfmail,Tmail=covtmail,reciver=covreciver)
        covermessageResponse=llm.invoke(covermessage)
        parsedouput=parser.parse(covermessageResponse)
        jsonparsed=parsedouput.dict()
        jsonstring=json.dumps(jsonparsed)
        with open('coverletter.json', 'w') as json_file:
            json_file.write(jsonstring)
        display_message("Bot", parsedouput.body)
        messagetype=0
        coverletterstep=0
        return 'return to main menu'


def changetimeSender(user_message):
    global timetmail,timefmail,timesubject,timedate,messagetype,changetimestep
    if changetimestep==1:
        timetmail=user_message
    elif changetimestep==2:
        timefmail=user_message
    elif changetimestep==3:
        timesubject=user_message
    elif changetimestep==4:
        timedate=user_message
        changeMeeting=changeorsetupmeetingTemp.format(Temail=timetmail,femail=timefmail,subject=timesubject,date=timedate)
        changeMeetingResponse=llm.invoke(changeMeeting)
        parsedouput=parser.parse(changeMeetingResponse)
        jsonparsed=parsedouput.dict()
        jsonstring=json.dumps(jsonparsed)
        with open('changetime.json', 'w') as json_file:
            json_file.write(jsonstring)
        display_message("Bot", parsedouput.body)
        messagetype=0
        changetimestep=0
        return 'return to main menu'
def changetimeRespond():
    global changetimestep
    if changetimestep==0:
        changetimestep=1
        return "what is the recipient email"
    elif changetimestep==1:
        changetimestep=2
        return "what is the sender email"
    elif changetimestep==2:
        changetimestep=3
        return "what is the email subject"
    elif changetimestep==3:
        changetimestep=4
        return "what is the date of the meeting"
def send_message():
    user_message = entry_message.get()
    global messagetype
    global subject, tmail, fmail, sender, receiver
    global generatingMailSteps

    if user_message.strip():
        display_message("User", user_message)  # Display user message first
        entry_message.delete(0, END)

        # Here you would include the logic to generate the response
        if messagetype == 1:
            if generatingMailSteps == 1:
                fmail = user_message
            elif generatingMailSteps == 2:
                tmail = user_message
            elif generatingMailSteps == 3:
                subject = user_message
            elif generatingMailSteps == 4:
                sender = user_message
            elif generatingMailSteps == 5:
                receiver = user_message
                newmail = writingFromScratchTemp.format(
                    subject=subject, Temail=tmail, femail=fmail, sender=sender, reciver=receiver)
                newMailResponse = llm.invoke(newmail)
                parsedouput = parser.parse(newMailResponse)
                jsonparsed=parsedouput.dict()
                jsonstring=json.dumps(jsonparsed)
                with open('generatemail.json', 'w') as json_file:
                    json_file.write(jsonstring)
                display_message("Bot", parsedouput.body)
                messagetype = 0
                generatingMailSteps = 0
        elif messagetype == 2:
            returnedmodvar = modifyEmailSend(user_message)
            if returnedmodvar is not None:
                user_message = returnedmodvar
        elif messagetype == 3:
            returnedcovervar = coverletterSender(user_message)
            if returnedcovervar is not None:
                user_message = returnedcovervar
        elif messagetype == 4:
            returnedtimevar = changetimeSender(user_message)
            if returnedtimevar is not None:
                user_message = returnedtimevar

        response = generate_response(user_message)
        display_message("Bot", response)

def display_message(sender, message):
    chat_area.config(state=NORMAL)
    # chat_area.insert(END, f"{sender}: {message}\n")
    if sender == "User":      
        message_frame = Frame(chat_area, bg="#235ded", bd=0)
        Label(message_frame, text=message, bg="#235ded", fg="white", font=("Helvetica", 12), wraplength=1250, anchor="w",justify=LEFT).pack(padx=5, pady=5, side=RIGHT)
        chat_area.window_create(END, window=message_frame)
        chat_area.insert(END, "\n")
        chat_area.tag_add("user_tag", "end-2l", "end-1l")  # This helps to tag the last inserted message.
        chat_area.tag_configure("user_tag", justify='right')  # Align the user's message to the right
    else:
        message_frame = Frame(chat_area, bg="#414141", bd=0)
        Label(message_frame, text=message, bg="#414141", fg="#C6CCD1", font=("Helvetica", 12), wraplength=1250, anchor="w", justify=LEFT).pack(padx=5, pady=5, anchor="w")
        chat_area.window_create(END, window=message_frame)
        chat_area.insert(END, "\n")
    chat_area.config(state=DISABLED)
    chat_area.yview(END)

def generate_response(message):
    global messagetype
    global generatingMailSteps
    global modifyMailstep
    global coverletterstep
    if message=='exist':
        messagetype=0
        coverConversation.memory.clear()
        

    if messagetype==0:

        requesttypeString=router.invoke({"request": message})
        if requesttypeString.templatetouse=="chooseAtask":
            messagetype=0
            return "I can help you with generating emails, modifying emails, generating cover letters, getting company information, and changing meeting times"
        elif requesttypeString.templatetouse=="generateEmail":
            messagetype=1
            generatingMailSteps=1
            return "what is your email"
        elif requesttypeString.templatetouse=="modifyEmail":
            messagetype=2
            modifyMailstep=0
            return modifyMailrespond()
        elif requesttypeString.templatetouse=="generatecoverletter":
            messagetype=3
            coverletterstep=1
            return "what is the job you are applying for"
        elif requesttypeString.templatetouse=="getcompanyinformation":
            display_message("Bot", output_info(message))
            return 'can I help you with anything else'
        elif requesttypeString.templatetouse=="changetime":
            messagetype=4
            # changetimestep=0
            return changetimeRespond()
        else:
            return "!!!!"
    if messagetype==1:
        if generatingMailSteps==0:
            generatingMailSteps=1
            return "what is your email"
        elif generatingMailSteps==1:
            generatingMailSteps=2
            return "what is the recipient email"
        elif generatingMailSteps==2:
            generatingMailSteps=3
            return "what is the email subject"
        elif generatingMailSteps==3:
            generatingMailSteps=4
            return "what is the name of the sender"
        elif generatingMailSteps==4:
            generatingMailSteps=5
            return "what is the name of the recipient"
    elif messagetype==2:
        return modifyMailrespond()
    elif messagetype==3:
        return coverletterRespond()
    elif messagetype==4:
        return changetimeRespond()
    return response

frame = cm.CTkFrame(master=root)
frame.pack(pady=10, padx=10, fill="both", expand=True)

label = cm.CTkLabel(master=frame, text="Elegant Writer", font=("Helvetica", 20, "bold"))
label.pack(pady=10, padx=10)

chat_area = Text(frame, wrap=WORD, state=DISABLED, font=("Helvetica", 12), bg="#2b2b2b", fg="white", insertbackground="white")
chat_area.pack(pady=10, padx=10, fill=BOTH, expand=True)

entry_frame = cm.CTkFrame(master=frame, fg_color=None)
entry_frame.pack(pady=10, padx=10, fill=X, expand=False)

entry_message = cm.CTkEntry(master=entry_frame, placeholder_text="Type your message here...", font=("Helvetica", 12))
entry_message.pack(side=LEFT, pady=5, padx=5, fill=X, expand=True)
entry_message.bind("<Return>", lambda event: send_message())

send_button = cm.CTkButton(master=entry_frame, text="Send", command=send_message, fg_color="#235ded", font=("Helvetica", 12, "bold"))
send_button.pack(side=RIGHT, pady=10, padx=10)
display_message("Bot", "Hey there! How can I assist you today?")
display_message("Bot", "I can help you with generating emails, modifying emails, generating cover letters, getting company information, and changing meeting times")

root.mainloop()

# %% [markdown]
# 


