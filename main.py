from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser, JsonOutputParser
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_community.vectorstores import Chroma
from pydantic import BaseModel, Field
from typing import Annotated, List, Optional
from typing import Dict
from dotenv import load_dotenv
import json
import time

load_dotenv()


file_path = "/Users/alidhaga/Desktop/IFRA_RAG_project/IFRA_standards"

loader = DirectoryLoader(
    file_path,
    glob="**/*.pdf",
    loader_cls=PyMuPDF4LLMLoader
)

documents = loader.lazy_load()

tables = []

for table in documents:
    tables.append(table.page_content)


class Table(BaseModel):
    name : str = Field(description="Name of the ingredient")
    CAS_NO : str = Field(description="Includes the unique CAS (Chemical Abstracts Servic) number of the chemical ingredient", default="Unknown")
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
for doc in tables[:3]:
    try:
        result = data_chain.invoke({'data' : doc})
        extracted_data.append(result)
        time.sleep(4)
    except Exception as e:
        print(f"Failed document : {e}")
        time.sleep(4)

print(extracted_data)
prompt = PromptTemplate(
    template="""You are an IFRA compliance checker, that receives a query and calculates whether it is IFRA compliant or not from the given context
    query:{query} \n\n  context: {retrived_data} \n\n

    If the query isn't related to IFRA standards, strictly reply with 'Query doesn't seems to be related to IFRA standards, please try again'!
    Do not create your own answers , reply only on the basics on the given contextual data""",
    input_variables=['query' , 'retrived_data']

)

    
"""def get_metadata(record: Dict, metadata: Dict) -> Dict:
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
    docs.append(Document(page_content=page_content, metadata=doc_metdata))"""

"""

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

