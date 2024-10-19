import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_index.core import SimpleDirectoryReader, SummaryIndex, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.llms import CustomLLM, CompletionResponse, CompletionResponseGen, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from transformers import BartTokenizer, BartForConditionalGeneration

# Load the model and tokenizer
model_name = "unsloth/llama-3-8b-bnb-4bit"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# Create a text generator pipeline
text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=128
)

# Define the custom LLM class
class OurLLM(CustomLLM):
    context_window: int = 3900
    num_output: int = 128
    model_name: str

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        generated = text_generator(prompt, max_new_tokens=self.num_output)[0]['generated_text']
        return CompletionResponse(text=generated)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, **kwargs: Any
    ) -> Generator[CompletionResponseGen, None, None]:
        response = ""
        generated = text_generator(prompt, max_new_tokens=self.num_output)[0]['generated_text']
        for token in generated:
            response += token
            yield CompletionResponseGen(text=response, delta=token)

# Create an instance of the custom LLM
llm = OurLLM(context_window=3900, num_output=128, model_name="lora_model")

# Create a Streamlit app
st.title("VerbaSumm Multilingual Summarization and Q&A")

# Add a file uploader
uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf", "docx"])

# Add a text input for the query
query = st.text_input("Enter a query")

# Add a button to generate the response
if st.button("Generate Response"):
    # Load the uploaded file
    if uploaded_file is not None:
        file_path = uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Create a directory reader
        directory_path = "/content/my_directory"
        os.makedirs(directory_path, exist_ok=True)
        documents = SimpleDirectoryReader(directory_path).load_data()

        # Create a summary index
        index = SummaryIndex.from_documents(documents)

        # Create a vector store index
        index2 = VectorStoreIndex.from_documents(documents)

        # Summarize the documents
        document_texts = [doc.text for doc in documents]
        summarized_texts = summarize_texts(document_texts)

        # Create a query engine
        query_engine = index2.as_query_engine()

        # Generate the response
        response = query_engine.query(query)
        st.write("Response:", response)

    else:
        st.write("Please upload a file")

# Add a button to summarize the documents
if st.button("Summarize Documents"):
    # Summarize the documents
    document_texts = [doc.text for doc in documents]
    summarized_texts = summarize_texts(document_texts)

    # Display the summarized texts
    st.write("Summarized Texts:")
    for text in summarized_texts:
        st.write(text)

def summarize_texts(input_texts, batch_size=8):
    summaries = []
    for i in range(0, len(input_texts), batch_size):
        batch = input_texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", max_length=1024, truncation=True, padding=True)
        summary_ids = model.generate(inputs['input_ids'], max_length=100, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
        batch_summaries = [tokenizer.decode(g, skip_special_tokens=True) for g in summary_ids]
        summaries.extend(batch_summaries)
    return summaries