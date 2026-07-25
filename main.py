from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser, JsonOutputParser
from langchain_core.exceptions import OutputParserException
import pymupdf4llm
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from pydantic import BaseModel, Field
from typing import Annotated, List, Optional
from typing import Dict
from dotenv import load_dotenv
from pathlib import Path
import sys
import json
import time

load_dotenv()

base_dir = Path("/Users/alidhaga/Desktop/IFRA_RAG_project/IFRA_standards")
file_paths= list(base_dir.rglob('**/*.pdf'))

contents = []
for file_path in file_paths[:10]:
    contents.append(pymupdf4llm.to_markdown(file_path))

class Table(BaseModel):
    name : str = Field(description="Name of the ingredient")
    CAS_NO : List[str] = Field(description="Extract only the numeric CAS numbers (e.g., 505-57-7). Explicitly ignore and remove any boilerplate sentences such as The scope of this Standard... If multiple numbers exist, separate them with commas")
    synonyms : Optional[List[str]] = Field(description="Should contain other name/names of the ingredient")
    type : List[str] = Field(description="should contain the type of ingredient it is, it can contain one or two values from the following values : RESTRICTION, PROHIBITION , SPECIFICATION")
    limits : Dict[str , Optional[float]] = Field(description="Includes the data of usable limtis of the ingredient, if the type contains PROHIBITION, then all the values across all the category should be None, the categories include Category 1: Products applied to the lips (lip balms, lipsticks).Category 2: Deodorants of all types.Category 3: Products applied to the face/body using the fingertips (eye products, facial make-up, nail care).Category 4: Products related to fine fragrances (eau de toilette, perfume, solid perfume, aftershave balms/creams).Category 5 (A, B, C, D): Leave-on products applied to the body, face, hands, and babies (e.g., body lotions, face creams, baby powders, and oils).Category 6: Mouthwash and toothpaste.Category 7 (A, B): Hair treatments (hair sprays, styling aids, hair dyes, shampoos).Category 8: Intimate hygiene products (intimate wipes, intimate deodorant sprays).Category 9: Rinse-off products (soaps, liquid soaps, body wash, bath bombs, shaving cream).Category 10 (A, B): Household care products with hand contact (laundry detergents, hard surface cleaners, air fresheners).Category 11 (A, B): Products with intended skin contact but minimal transfer (feminine hygiene pads, baby diapers, tissues).Category 12: Products with no intentional skin contact (candles, plug-in air fresheners, incense)")


parser = PydanticOutputParser(pydantic_object=Table)
    
data_template = PromptTemplate(
    template="Generate values based on the given data {data}, strictly follow this schema {format_instructions} ",
    input_variables=['data'],
    partial_variables={'format_instructions' : parser.get_format_instructions()}
)

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
) 

data_chain = data_template | model | parser

extracted_data = []

i = 0
z = 10 # index value
for data in contents[:z]:
    if i <= z:
        while True:
            try:
                result = data_chain.invoke({'data' : data})
                extracted_data.append(result)
                json_list = [item.model_dump() for item in extracted_data]
                with open('extracted_data_list' , 'w' , encoding='utf-8') as f:
                    json.dump(json_list , f ,indent=4)
                i = i + 1
                break
            except OutputParserException as e:
                sys.exit(f"Pydantic error stopping the program, start again form index{i}")

            except Exception as e:
                print(f"Prolly rate limit error for {i}")
                time.sleep(30)
    else:
        print(f"All {i} data extracted successfully!")
            

prompt = PromptTemplate(
    template="""You are an IFRA compliance checker, that receives a query and calculates whether it is IFRA compliant or not from the given context
    query:{query} \n\n  context: {retrived_data} \n\n

    If the query isn't related to IFRA standards, strictly reply with 'Query doesn't seems to be related to IFRA standards, please try again'!
    Do not create your own answers , reply only on the basics on the given contextual data""",
    input_variables=['query' , 'retrived_data']
)

"""
def get_metadata(record: Dict, metadata: Dict) -> Dict:
    metadata['name'] = record.get('name')
    metadata['cas_number'] = record.get('cas_number')
    metadata['type'] = json.dumps(record.get('type'))
    metadata['specification_text'] = json.dumps(record.get('specification_text', "None"))
    metadata['limits'] = json.dumps(record.get('limits'))
    return metadata

docs = []
for chemical in data:
    page_content = f'NAME: {chemical['name']} CAS_NUMBER: {chemical['cas_number']} SYNONYMS: {', '.join(chemical['synonyms'])}'
    doc_metdata= get_metadata(chemical, {})
    docs.append(Document(page_content=page_content, metadata=doc_metdata))

embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2", output_dimensionality=70
)

vector_store = Chroma.from_documents(
        documents=tables,
        embedding=embedding_model,
        collection_name="ifra_standards"
)

vector_store_retriver = vector_store.as_retriever(search_kwargs={"k" : 1})
BM25_retriver = BM25Retriever.from_documents(documents=tables)

retriver = EnsembleRetriever(
    retrievers=[vector_store_retriver, BM25_retriver],
    weights=[0.8 , 0.2]
)

parser = JsonOutputParser()
"""
