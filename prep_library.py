import os
import sys
from pypdf import PdfReader # For reading PDFs
from langchain_text_splitters import RecursiveCharacterTextSplitter # For chunking text
from langchain_core.documents import Document # Standard data structure for chunks
from PIL import Image, UnidentifiedImageError # Pillow for handling images
import io # For handling image bytes
# import hashlib # <-- REMOVED
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
# --- Imports for Embeddings and FAISS ---
from langchain_huggingface import HuggingFaceEmbeddings # Embeddings model wrapper
from langchain_community.vectorstores import FAISS # FAISS vector store

# Set default encoding to UTF-8
try:
    # These might fail in some environments like basic terminals but work in others
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except AttributeError:
    # print("Warning: Could not reconfigure stdout/stderr encoding.")
    pass # Continue even if reconfiguration fails

# --- Initialize Image Captioning Model (Load once) ---
# Determine device: CUDA (GPU) if available, otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device for image captioning: {device}")
# Use float16 for faster inference/less memory on GPU, float32 on CPU
dtype = torch.float16 if device == "cuda" else torch.float32
caption_processor = None
caption_model = None
try:
    print("Loading image captioning model (Salesforce/blip-image-captioning-base)...")
    caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    # Load model onto the determined device and with the chosen dtype
    caption_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base", torch_dtype=dtype
    ).to(device)
    print("Image captioning model loaded successfully.")
except Exception as e:
    print(f"Error loading image captioning model: {e}")
    print("Image captioning features will be disabled.")


def process_pdfs_text(folder_path):
    """
    Reads all PDF files in a specified folder, extracts text, chunks it,
    and returns a list of LangChain Document objects with metadata.

    Args:
        folder_path (str): The path to the folder containing PDF files.

    Returns:
        list[Document]: A list of LangChain Document objects for text chunks.
    """
    all_pdf_docs = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    print(f"\nStarting PDF text processing in folder: {folder_path}")

    if not os.path.isdir(folder_path):
        print(f"Error: Folder not found at {folder_path}")
        return []

    try:
        # Get list of files, ignore potential subdirectories for now
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    except OSError as e:
        print(f"Error listing files in {folder_path}: {e}")
        return []

    processed_files = 0
    total_chunks = 0

    pdf_files_found = [f for f in files if f.lower().endswith(".pdf")]
    if not pdf_files_found:
        print("No PDF files found in this directory.")
        return []

    print(f"Found {len(pdf_files_found)} PDF files to process for text.")

    for filename in pdf_files_found:
        pdf_path = os.path.join(folder_path, filename)
        print(f"    Processing text from: {filename}...")
        try:
            reader = PdfReader(pdf_path)
            pdf_text = ""
            page_map = [] # To map character index to page number roughly

            for page_num, page in enumerate(reader.pages):
                start_index = len(pdf_text)
                page_content = page.extract_text()
                if page_content:
                    pdf_text += page_content + "\n" # Add a newline between pages
                    page_map.append((start_index, page_num + 1))
                else:
                    # Keep track of pages with no text if needed for page mapping consistency
                    page_map.append((start_index, page_num + 1))
                    # print(f"      Warning: No text extracted from page {page_num + 1} of {filename}")


            if pdf_text.strip(): # Check if any text was extracted after stripping whitespace
                # Use the splitter's create_documents method for potential metadata handling
                # Need to wrap pdf_text in a list for create_documents
                temp_doc = Document(page_content=pdf_text)
                text_chunks_docs = text_splitter.split_documents([temp_doc])

                file_chunks = 0
                for i, doc_chunk in enumerate(text_chunks_docs):
                    # Find which page this chunk likely started on
                    # This requires tracking character offsets, which split_documents doesn't easily provide.
                    # A simpler approximation: use the midpoint or start of the chunk content
                    # to find its position in the original text and map to page.
                    # For now, we'll keep the simpler method based on page markers if needed,
                    # or just tag the source file. Let's keep it simple for now.

                    metadata = {
                        "source": filename,
                        "type": "pdf_text",
                        # "page": approx_page # Page number estimation can be complex and inaccurate
                    }
                    # Update the metadata of the chunk created by split_documents
                    doc_chunk.metadata.update(metadata)
                    all_pdf_docs.append(doc_chunk)
                    file_chunks += 1

                print(f"      Extracted {file_chunks} text chunks from {filename}.")
                total_chunks += file_chunks
                processed_files += 1
            else:
                print(f"      Warning: No text could be extracted from {filename} (it might be image-based or empty).")

        except FileNotFoundError:
            print(f"      Error: File not found at {pdf_path}. Skipping.")
        # Catch specific pypdf errors if known, otherwise general exception
        except Exception as e:
            print(f"      Error processing text in {filename}: {e}")

    print(f"\nFinished PDF text processing.")
    print(f"Successfully processed {processed_files} PDF files for text.")
    print(f"Total text chunks created: {total_chunks}")

    return all_pdf_docs


# --- Set of known logo hashes to ignore ---
# --- All hash logic removed as requested ---


def process_pdf_images(folder_path):
    """
    Reads all PDF files, extracts images, generates captions, and returns
    a list of Document objects for these captions.
    """
    all_image_docs = []

    # --- 'hashes_captioned_this_run' set has been REMOVED to process all repeated non-logo images ---

    if caption_processor is None or caption_model is None:
        print("\nImage captioning model not loaded. Skipping image processing.")
        return []

    print(f"\nStarting PDF image processing in folder: {folder_path}")

    if not os.path.isdir(folder_path):
        print(f"Error: Image source folder not found at {folder_path}")
        return []

    try:
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    except OSError as e:
        print(f"Error listing files in {folder_path}: {e}")
        return []

    processed_files_count = 0
    total_images_processed = 0
    total_captions_generated = 0
    # total_logos_skipped = 0 # <-- REMOVED

    pdf_files_found = [f for f in files if f.lower().endswith(".pdf")]
    if not pdf_files_found:
        print("No PDF files found in this directory for image processing.")
        return []

    print(f"Found {len(pdf_files_found)} PDF files to check for images.")

    for filename in pdf_files_found:
        pdf_path = os.path.join(folder_path, filename)
        print(f"    Processing images from: {filename}...")
        try:
            reader = PdfReader(pdf_path)
            images_in_file = 0
            captions_in_file = 0
            file_processed_flag = False # Flag if we attempt processing images in this file

            for page_num, page in enumerate(reader.pages):
                image_count_on_page = 0
                # Check if page has images and iterate through them
                if hasattr(page, 'images') and page.images:
                    file_processed_flag = True # Mark that we are processing images for this file
                    for image_index, image_file_object in enumerate(page.images):
                        image_count_on_page += 1
                        try:
                            image_bytes = image_file_object.data

                            # --- BEGIN HASH CHECK ---
                            # --- All hash logic removed ---
                            
                            # [NEW] Add a debug print to see what's being captioned.
                            # --- [DEBUG] print statement removed ---
                            
                            # Attempt to open the image using Pillow
                            image = Image.open(io.BytesIO(image_bytes))

                            # Basic check for very small images which might be icons/spacers
                            min_dimension = 50 # Ignore images smaller than 50x50 pixels (adjustable)
                            if image.width < min_dimension or image.height < min_dimension:
                                # print(f"      Skipping very small image ({image.width}x{image.height}) on page {page_num+1}.")
                                continue

                            # Ensure image is in RGB format
                            if image.mode != "RGB":
                                image = image.convert("RGB")

                            # --- Generate Caption ---
                            # Prepare image
                            inputs = caption_processor(images=image, return_tensors="pt").to(device, dtype)

                            # Generate caption with torch.no_grad() for efficiency
                            with torch.no_grad():
                                output_ids = caption_model.generate(**inputs, max_new_tokens=50) # Use max_new_tokens
                            caption = caption_processor.decode(output_ids[0], skip_special_tokens=True)

                            # Check if caption is meaningful (basic check)
                            if caption and not caption.isspace() and len(caption) > 5:
                                metadata = {
                                    "source": filename,
                                    "page": page_num + 1,
                                    "image_index_on_page": image_index,
                                    "type": "image_caption"
                                }
                                doc = Document(page_content=caption.strip(), metadata=metadata)
                                all_image_docs.append(doc)
                                captions_in_file += 1
                            else:
                                # print(f"      Warning: Empty or very short caption generated for an image on page {page_num+1} of {filename}")
                                pass # Suppress warning for empty captions if desired

                        except UnidentifiedImageError:
                            print(f"      Warning: Could not identify image format on page {page_num+1}, image index {image_index} in {filename}. Skipping.")
                        except ImportError as ie:
                            # Specific check for common decoder issues
                            if 'jpeg' in str(ie).lower() or 'decoder' in str(ie).lower() or 'png' in str(ie).lower():
                                print(f"      Warning: Missing/corrupt image decoder for image on page {page_num+1}, index {image_index} in {filename}. Skipping. (Ensure Pillow dependencies are correct)")
                            else:
                                print(f"      ImportError processing image on page {page_num+1} of {filename}: {ie}")
                        except ValueError as ve:
                             print(f"      ValueError processing image on page {page_num+1} of {filename} (often memory related): {ve}")
                        except Exception as img_e:
                            print(f"      Unexpected Error processing image on page {page_num+1} of {filename}: {img_e}")


                if image_count_on_page > 0:
                    images_in_file += image_count_on_page
            # End of page loop

            if images_in_file > 0:
                print(f"      Found {images_in_file} potential image objects, generated {captions_in_file} captions for {filename}.")
                total_images_processed += images_in_file
                total_captions_generated += captions_in_file
            #else:
            #   print(f"      No image objects found or processed in {filename}.")

            if file_processed_flag: # Count file if we attempted image processing
                processed_files_count += 1


        except FileNotFoundError:
            print(f"      Error: File not found at {pdf_path}. Skipping.")
        # Catch specific pypdf errors if known, otherwise general exception
        except Exception as e:
            print(f"      Error opening or reading {filename} for image processing: {e}")

    # End of file loop

    print(f"\nFinished PDF image processing.")
    print(f"Attempted to process images in {processed_files_count} PDF files.")
    print(f"Total potential image objects encountered: {total_images_processed}")
    # print(f"Total known/repeated logos skipped: {total_logos_skipped}") # <-- REMOVED
    print(f"Total valid image captions generated: {total_captions_generated}")

    return all_image_docs


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Configuration ---
    # Define the folder containing your PDF files
    pdf_folder = r"C:\Users\saksh\gpai\data\PDFs\DSA" # <<< YOUR PDF FOLDER PATH HERE
    # Define the folder where the FAISS index will be saved
    faiss_index_path = "faiss_index_uni_buddy"
    # Define the embedding model to use
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Initialization ---
    all_processed_docs = [] # List to hold all documents (text + captions)
    pdf_files_exist_in_folder = False # Flag to check if processing is needed

    # --- Step 1: Process PDF Text ---
    print("\n=== Processing PDF Text ===")
    if not os.path.exists(pdf_folder):
        print(f"Error: PDF Directory not found: {pdf_folder}")
    else:
        try:
            # Check if there are any PDF files before proceeding
            pdf_files_exist_in_folder = any(f.lower().endswith(".pdf") for f in os.listdir(pdf_folder) if os.path.isfile(os.path.join(pdf_folder, f)))
            if not pdf_files_exist_in_folder:
                 print(f"No PDF files found in {pdf_folder}.")
            else:
                print("-" * 20)
                # Call the text processing function
                pdf_text_docs = process_pdfs_text(pdf_folder)
                all_processed_docs.extend(pdf_text_docs)
                print("-" * 20)
        except OSError as e:
            print(f"Error reading PDF directory {pdf_folder}: {e}")

    # --- Step 2: Process Images within PDFs ---
    print("\n=== Processing Images within PDFs ===")
    if not os.path.exists(pdf_folder):
         pass # Error already printed above
    elif not pdf_files_exist_in_folder:
         pass # Message already above
    else:
        print("-" * 20)
        pdf_image_docs = process_pdf_images(pdf_folder) # Call the image processing function
        all_processed_docs.extend(pdf_image_docs)
        print("-" * 20)
        # Note: Error handling for image processing is inside the function

    # --- Step 3: Create Embeddings and FAISS Index ---
    print("\n=== Creating Embeddings and FAISS Index ===")
    if not all_processed_docs:
        print("No documents were processed (text or images), cannot create index.")
    else:
        print(f"Total documents to index: {len(all_processed_docs)}")
        try:
            # --- Load Embedding Model ---
            # Determine device for embeddings (can be different from captioning)
            embedding_device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"\nLoading embedding model ({embedding_model_name})...")
            print(f"Using device for embeddings: {embedding_device}")

            # Initialize HuggingFace embeddings
            # model_kwargs handles device placement
            embeddings_model = HuggingFaceEmbeddings(
                model_name=embedding_model_name,
                model_kwargs={'device': embedding_device},
                encode_kwargs={'normalize_embeddings': True} # Normalize for better cosine similarity
            )
            print("Embedding model loaded successfully.")

            # --- Create FAISS Index ---
            print("\nCreating FAISS index from documents...")
            print("This step involves generating embeddings for all document chunks and may take a significant amount of time, especially on CPU...")

            # FAISS.from_documents handles the embedding calculation internally
            db = FAISS.from_documents(all_processed_docs, embeddings_model)
            print("FAISS index created successfully in memory.")

            # --- Save the Index Locally ---
            print(f"Saving FAISS index to folder: '{faiss_index_path}'")
            db.save_local(faiss_index_path)
            print("FAISS index saved successfully.")

        except Exception as e:
            print(f"!!! An error occurred during embedding creation or index saving: {e}")
            print("Index may not have been saved correctly.")


    # --- Final Summary ---
    print("\n" + "=" * 30)
    print("=== FINAL PROCESSING SUMMARY ===")
    print("=" * 30)
    if all_processed_docs:
        print(f"Total document chunks processed and added to index: {len(all_processed_docs)}")
        # Count types based on metadata
        text_count = sum(1 for doc in all_processed_docs if doc.metadata.get('type') == 'pdf_text')
        image_count = sum(1 for doc in all_processed_docs if doc.metadata.get('type') == 'image_caption') # <-- FIXED
        print(f"   - Text chunks: {text_count}")
        print(f"   - Image captions: {image_count}")

        # Check if index folder exists as final confirmation
        if os.path.exists(faiss_index_path) and os.path.exists(os.path.join(faiss_index_path, "index.faiss")):
             print(f"\nVector index successfully saved to folder: '{faiss_index_path}'")
             print("Ready for Phase 3 (Chatbot Application).")
        else:
             print("\n!!! ERROR: Vector index folder or 'index.faiss' file not found after saving process.")
             print("   Please review logs for errors during index creation or saving.")
    else:
        print("No documents were processed from any source.")
        print("Vector index was NOT created.")
        print("Please check your PDF folder path and ensure it contains processable PDF files.")
    print("=" * 30)

