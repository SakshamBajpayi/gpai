import os
import sys
# Add flush=True to ensure immediate output
print("--- Script Starting ---", flush=True)

try:
    from pypdf import PdfReader
    import hashlib
    from PIL import Image, UnidentifiedImageError
    import io
    print("--- Imports successful ---", flush=True)
except ImportError as e:
    print(f"!!! Import Error: {e}", flush=True)
    sys.exit("Exiting due to import error.")
except Exception as e:
    print(f"!!! Unexpected error during imports: {e}", flush=True)
    sys.exit("Exiting due to unexpected import error.")


# --- Configuration ---
# Put the path to ONE PDF file that definitely contains the logo
# pdf_to_check = r"C:\Users\saksh\gpai\data\PDFs\DSA\FILENAME_WITH_LOGO.pdf" # Replace with an actual PDF filename
pdf_to_check = r"C:\Users\saksh\gpai\data\PDFs\DSA\QueueImplementationLinkedList.pdf" # Example using the file you uploaded
print(f"--- PDF to check: {pdf_to_check} ---", flush=True)

image_hashes = {} # Dictionary to store hashes and example locations

print(f"Checking for images in: {os.path.basename(pdf_to_check)}", flush=True)

if not os.path.exists(pdf_to_check):
    print(f"Error: File not found at {pdf_to_check}", flush=True)
    sys.exit(1)
else:
    print(f"--- PDF file exists. Proceeding... ---", flush=True)


try:
    print("--- Attempting to open PDF with PdfReader... ---", flush=True)
    reader = PdfReader(pdf_to_check)
    print(f"--- PdfReader opened successfully. Found {len(reader.pages)} pages. ---", flush=True)

    # page_limit = len(reader.pages) # Process all pages
    # === LIMIT TO FIRST 5 PAGES FOR TESTING ===
    page_limit = 5
    print(f"--- Processing ONLY up to {page_limit} pages for faster testing... ---", flush=True) # DEBUG PRINT 7

    for page_num, page in enumerate(reader.pages):
        if page_num >= page_limit:
             print(f"--- Reached page limit ({page_limit}). Stopping page iteration. ---", flush=True)
             break # Stop if page limit is reached

        print(f"  Checking page {page_num + 1}...", flush=True)
        if hasattr(page, 'images') and page.images:
            print(f"    Found {len(page.images)} images object(s) on page {page_num + 1}.", flush=True)
            for image_index, img_obj in enumerate(page.images):
                print(f"      Processing image index {image_index}...", flush=True)
                try:
                    img_bytes = img_obj.data
                    print(f"        Got image bytes (length: {len(img_bytes)}). Calculating hash...", flush=True)
                    img_hash = hashlib.sha256(img_bytes).hexdigest()
                    print(f"        Calculated hash: {img_hash[:10]}...", flush=True)

                    # Try to get dimensions for identification help
                    dims = "N/A" # Default
                    try:
                        print("          Attempting to get image dimensions with PIL...", flush=True)
                        image = Image.open(io.BytesIO(img_bytes))
                        dims = f"{image.width}x{image.height}"
                        print(f"          Got dimensions: {dims}", flush=True)
                    except UnidentifiedImageError:
                        dims = "Unknown dimensions (UnidentifiedImageError)"
                        print("          Failed to get dimensions: UnidentifiedImageError", flush=True)
                    except Exception as dim_e:
                         dims = f"Error getting dimensions: {dim_e}"
                         print(f"          Failed to get dimensions: {dim_e}", flush=True)


                    # Store the hash and where we first saw it
                    if img_hash not in image_hashes:
                         print(f"        New hash found. Storing info.", flush=True)
                         image_hashes[img_hash] = {
                            "first_seen": f"Page {page_num + 1}, Image Index {image_index}",
                            "dimensions": dims
                            }
                    else:
                         print(f"        Duplicate hash. Ignoring.", flush=True)


                except Exception as e:
                    print(f"      !!! Error processing image index {image_index} on page {page_num + 1}: {e}", flush=True)
        # else:
        #     print(f"  No images found on page {page_num + 1}.", flush=True) # Optional


except Exception as e:
    print(f"!!! Error reading PDF or during page/image iteration: {e}", flush=True)

print("\n--- Unique Image Hashes Found ---", flush=True)
if not image_hashes:
    print("No images found or processed in the PDF.", flush=True)
else:
    # Sort by first appearance for clarity
    try:
        # Corrected sorting key extraction
        sorted_hashes = sorted(image_hashes.items(), key=lambda item: (int(item[1]['first_seen'].split(' ')[1].replace(',', '')), int(item[1]['first_seen'].split(' ')[-1])))
        for img_hash, info in sorted_hashes:
            print(f"Hash: {img_hash[:10]}... (Full: {img_hash})", flush=True)
            print(f"  Dimensions: {info['dimensions']}", flush=True)
            print(f"  First Seen: {info['first_seen']}", flush=True)
            print("-" * 10, flush=True)
    except Exception as sort_e:
        print(f"!!! Error sorting or printing hashes: {sort_e}", flush=True)


print(f"\nFound {len(image_hashes)} unique image hashes.", flush=True)
print("Identify the hash(es) corresponding to the university logo.", flush=True)
print("--- Script End ---", flush=True)

