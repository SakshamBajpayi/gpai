import os
import sys
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import traceback 

# --- Configuration ---
FAISS_INDEX_PATH = "faiss_index_uni_buddy"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# LLM Configuration - Using Hugging Face model
LLM_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"

# Set default encoding to UTF-8
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except AttributeError:
    pass

# --- IMPORTANT: Upgrade Check ---
# If you run into cache errors, please run this command once:
# pip install -U "transformers" accelerate safetensors
# Then delete the model's custom cache folder here:
# Remove-Item -Recurse -Force "$env:USERPROFILE\.cache\huggingface\modules\transformers_modules\microsoft_Phi_hyphen_3*"


def initialize_components():
    """
    Loads and initializes all necessary components for the RAG chatbot.
    Returns:
        A tuple of (retriever, llm, prompt) or (None, None, None) if setup fails.
    """
    
    # --- 1. Check if FAISS Index Exists ---
    if not os.path.exists(FAISS_INDEX_PATH):
        print(f"!!! FATAL ERROR: FAISS index folder not found at '{FAISS_INDEX_PATH}' !!!")
        print("Please run the 'prep_library.py' script first to create the index.")
        return None, None, None

    try:
        # --- 2. Load Embedding Model ---
        print("Loading embedding model...")
        embedding_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        embeddings_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': embedding_device},
            encode_kwargs={'normalize_embeddings': True}
        )
        print(f"Embedding model loaded successfully on {embedding_device}.")

        # --- 3. Load FAISS Index ---
        print(f"Loading FAISS index from '{FAISS_INDEX_PATH}'...")
        db = FAISS.load_local(
            FAISS_INDEX_PATH, 
            embeddings_model, 
            allow_dangerous_deserialization=True
        )
        retriever = db.as_retriever(search_kwargs={"k": 5})
        print("FAISS index loaded successfully.")

        # --- 4. Load Large Language Model (LLM) ---
        print(f"Loading LLM: {LLM_MODEL_ID}")
        print("This may take a few minutes on first run (downloading model)...")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype = torch.bfloat16 if device == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16 if device == 'cuda' else torch.float32
        
        print(f"Using device: {device}, dtype: {dtype}")

        # --- IMPORTANT: Removing trust_remote_code to use built-in Phi-3 support ---
        # This resolves the seen_tokens/get_max_length/IndexError issues.
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID, trust_remote_code=False) 
        
        model_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": False, # Crucial: forces using the compatible built-in class
            "attn_implementation": "eager"  
        }

        if device == 'cuda':
            print("Attempting to load model on GPU with 'device_map=auto'...")
            model_kwargs["device_map"] = "auto"
            model_kwargs["low_cpu_mem_usage"] = True
        else:
            print("Loading model on CPU...")
            pass 
            
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_ID,
            **model_kwargs
        )
        
        if device == 'cpu':
            print("Moving model to CPU...")
            model = model.to(device)

        # We disable use_cache here as a final measure, though it should be handled by model.generate
        try:
            model.config.use_cache = False
        except Exception:
            pass
        # --- End Model Loading ---


        text_gen_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=1024,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            model_kwargs={"use_cache": False, "return_dict_in_generate": True}
        )
        
        llm_pipeline = text_gen_pipeline
        print(f"LLM loaded successfully on {device}.")


        # --- 5. Create the RAG Prompt Template ---
        prompt_template_str = (
            "<|system|>\n"
            "You are an expert assistant for Data Structures and Algorithms. "
            "Use the following retrieved context (which includes text and image captions from lecture slides) "
            "to answer the user's question accurately and concisely. "
            "If you don't know the answer from the context, "
            "say 'I do not have that information in my knowledge base.'\n\n"
            "CONTEXT:\n{context}\n"
            "----------------\n<|end|>\n"
            "<|user|>\n{input}<|end|>\n"
            "<|assistant|>"
        )

        prompt = PromptTemplate(
            template=prompt_template_str, input_variables=["context", "input"]
        )

        print("\n" + "="*50)
        print("✓ Chatbot is Ready!")
        print("="*50)
        print("Type 'exit' or 'quit' to end the session.\n")
        
        return retriever, llm_pipeline, prompt

    except Exception as e:
        print(f"\n!!! An error occurred during initialization !!!")
        print(f"Error: {e}")
        traceback.print_exc()
        return None, None, None

def main():
    """
    Main chat loop.
    """
    retriever, llm_pipeline, prompt = initialize_components()
    if not all([retriever, llm_pipeline, prompt]):
        print("\nFailed to initialize. Exiting.")
        return

    while True:
        try:
            query = input("\n❓ Ask your question: ")
            if query.lower() in ['exit', 'quit']:
                print("\n👋 Goodbye!")
                break
            if not query.strip():
                continue

            print("\n🤔 Thinking...")
            
            # --- Manual RAG chain logic ---
            
            # 1. Retrieve: Get relevant documents
            docs = retriever.invoke(query)
            
            # 2. Augment: Format the context
            context_str = "\n\n".join([doc.page_content for doc in docs])
            
            # 3. Format the prompt
            formatted_prompt = prompt.format(context=context_str, input=query)
            
            # 4. Generate: Invoke the LLM pipeline
            response_list = llm_pipeline(formatted_prompt)
            raw_answer = response_list[0]['generated_text']
            
            # 5. Parse: Clean the model's output
            answer = raw_answer.split("<|assistant|>", 1)[-1].strip() # Use split(..., 1) for safety
            
            # --- Print the Answer ---
            print("\n" + "="*50)
            print("📝 Answer:")
            print("="*50)
            print(answer)
            
            # --- Print Sources with Page Numbers ---
            print("\n📚 Sources:")
            print("-"*50)
            sources_with_pages = []
            for doc in docs:
                source = doc.metadata.get('source', 'Unknown')
                page = doc.metadata.get('page')
                doc_type = doc.metadata.get('type', '')
                
                if page:
                    source_str = f"{source} (page {page})"
                else:
                    source_str = source
                    
                if doc_type == 'image_caption':
                    source_str += " [Image]"
                    
                sources_with_pages.append(source_str)
            
            unique_sources = list(dict.fromkeys(sources_with_pages))
            if unique_sources:
                for i, source in enumerate(unique_sources, 1):
                    print(f"[{i}] {source}")
            else:
                print("No specific sources found in context.")
            print("="*50)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            traceback.print_exc()
            print("\nPlease try again.")

if __name__ == "__main__":
    main()
