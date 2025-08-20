import tempfile
import fitz
import asyncio
import os
import cv2
import json
import uuid
import numpy as np
from datetime import datetime
from PIL import Image, ImageEnhance
from io import BytesIO
from pathlib import Path
from openai import OpenAI
from gptprocesses import try_process_image
# Simulated PDF processing function
async def process_pdf(uploaded_file, websocket, filename):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file)
            tmp_path = tmp_file.name

        page_count = get_pdf_page_count(tmp_path)
        await websocket.send_text(f"📊 **Pages:** {page_count}")

        results_dir = create_results_directory()
                    
                    # Process multi-page PDF  
        result = await process_multi_page_pdf(tmp_path, filename, websocket)
        
        

        # await websocket.send_text(result_json)
        # Clean up temp PDF file
        os.unlink(tmp_path)
        
        if result:
            
        #     # Store result in session state
        #     st.session_state.processed_result = result
            
        #     # Show processing summary
            pdf_info = result.get('pdf_info', {})
            await websocket.send_text(f"✅ **PDF processed successfully!**")
            await websocket.send_text(f"📊 **Summary:** {pdf_info.get('successful_pages', 0)}/{pdf_info.get('total_pages', 0)} pages processed")
            
        #     # Save results
        
        #             # Save individual file
            try:

                individual_file, enhanced_result = save_result_to_file(
                    result, filename, results_dir
                )

        #                 # Append to master results
                master_file, total_count = append_to_master_results(
                    result, filename, results_dir
                )
                await websocket.send_json({"result" : {"text" : enhanced_result} })
                await websocket.send_text(f"💾 **Results automatically saved:**")
                await websocket.send_text(f"📁 **Individual file:** `{os.path.basename(individual_file)}`")
                await websocket.send_text(f"📋 **Master file:** `all_results.json` (Total: {total_count} documents)")
                    
            except Exception as e:
                websocket.send_text(f"❌ Failed to save results: {e}")
                    
        else:
            websocket.send_text("❌ Failed to process PDF")
                       

        # Here, implement your actual PDF processing logic, e.g. using PyMuPDF, PyPDF2, etc.
         # Simulated result
        
        # Return processing result
        return {
            "status": "success",
        }
    except Exception as e:
        websocket.send_text(f"Error {e}")


async def process_image(uploaded_file, websocket, filename):
        try:
    # IMAGE PROCESSING (your existing code)
                image = Image.open(BytesIO(uploaded_file))
                                
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{filename.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file)
                    tmp_path = tmp_file.name
                
                # Create results directory
                results_dir = create_results_directory()
                
                # Process with smart retry
                result = await process_invoice_with_retry(tmp_path,websocket)
                
                # Clean up temp file
                os.unlink(tmp_path)
                
                if result:
                    # Store result in session state
                    # st.session_state.processed_result = result
                    
                    # Show quality assessment
                    if "quality_assessment" in result:
                        quality = result["quality_assessment"]
                        
                        if quality.get("quality_too_poor", False):
                            websocket.send_text("⚠️ **Image quality was initially poor - preprocessing was applied**")
                        
                        await websocket.send_text(f"📊 **Quality Assessment:**")
                        await websocket.send_text(f"- **Readability:** {quality.get('readability_score', 'N/A')}")
                        await websocket.send_text(f"- **Can extract data:** {quality.get('can_extract_data', 'N/A')}")
                        
                        if quality.get("quality_issues"):
                            await websocket.send_text("**Quality issues detected:**")
                            for issue in quality["quality_issues"]:
                                await websocket.send_text(f"  - {issue}")
                    
                    # Automatically save to JSON files
                    
                    try:
                        # Save individual file
                        individual_file, enhanced_result = save_result_to_file(
                            result, filename, results_dir
                        )
                        
                        # Append to master results
                        master_file, total_count = append_to_master_results(
                            result, filename, results_dir
                        )
                        
                        # Success messages
                        await websocket.send_json({"result" : {"text" : enhanced_result} })
                        await websocket.send_text(f"💾 **Results automatically saved:**")
                        await websocket.send_text(f"📁 **Individual file:** `{os.path.basename(individual_file)}`")
                        await websocket.send_text(f"📋 **Master file:** `all_results.json` (Total: {total_count} documents)")
                        await websocket.send_text(f"📂 **Location:** `{results_dir}/`")
                        
                    except Exception as e:
                        await websocket.send_text(f"❌ Failed to save results: {e}")
                        await websocket.send_text("✅ Invoice processed successfully (but not saved)")
                else:
                    await websocket.send_text("❌ Failed to process invoice even after preprocessing")

        except Exception as e:
            websocket.send_text(f"Error {e}")

def get_pdf_page_count(pdf_path):
    """Get number of pages in PDF"""
    try:
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        doc.close()
        return page_count
    except Exception as e:
        return 0
    
def create_results_directory():
    """Create results directory if it doesn't exist"""
    results_dir = "resultjson"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    return results_dir

async def process_multi_page_pdf(pdf_path, uploaded_filename, ws):
    """Process multi-page PDF with smart retry for each page"""
    # Get page count
    page_count = get_pdf_page_count(pdf_path)
    
    if page_count == 0:
        await ws.send_text("❌ Invalid PDF or no pages found")
        return None
    print("Page cnt",page_count)
    await ws.send_text(f"📄 **PDF detected with {page_count} pages**")
    
    # Convert PDF to images
    # with st.spinner('🔄 Converting PDF pages to images...'):
    #NEED TO IMPLEMENT SPINNER
    try:
        image_paths = pdf_to_images(pdf_path)
    except Exception as e:
        await ws.send_text("❌ Failed to convert PDF pages")
    
    
    
    # ADD DEBUG INFO
    await ws.send_text(f"🖼️ **Successfully converted {len(image_paths)} pages to images**")
    
    # Process each page
    all_page_results = []
    
    for i, image_path in enumerate(image_paths, 1):
        await ws.send_text(f"🔄 **Processing Page {i}/{page_count}...**")
        
        # Process with smart retry
        page_result =await  process_invoice_with_retry(image_path, ws)
        print(page_result)
        if page_result:
            # ADD DEBUG: Show what was extracted from this page
            vendor = page_result.get('invoice_header', {}).get('vendor_name', 'N/A')
            line_items_count = len(page_result.get('line_items', []))
            await ws.send_text(f"✅ Page {i} processed - Vendor: {vendor}, Line Items: {line_items_count}")
            
            # Add page metadata
            page_result['page_info'] = {
                'page_number': i,
                'total_pages': page_count,
                'source_pdf': uploaded_filename,
                'page_image': Path(image_path).name
            }
            all_page_results.append(page_result)
        else:
            await ws.send_text(f"⚠️ Page {i} processing failed")
            # Add failed page placeholder
            all_page_results.append({
                'page_info': {
                    'page_number': i,
                    'total_pages': page_count,
                    'source_pdf': uploaded_filename,
                    'processing_failed': True
                },
                'error': 'Page processing failed'
            })
    
    # ADD DEBUG: Show combination results
    await ws.send_text(f"📊 **Total processed results: {len(all_page_results)}**")
    
    # Clean up temporary image files
    for image_path in image_paths:
        try:
            os.unlink(image_path)
        except:
            pass
    
    # Combine results
    combined_result = combine_pdf_page_results(all_page_results, uploaded_filename)
    
    # ADD DEBUG: Show combined line items count
    combined_line_items = len(combined_result.get('combined_data', {}).get('line_items', []))
    await ws.send_text(f"🔗 **Combined line items: {combined_line_items}**")
    return combined_result

def pdf_to_images(pdf_path, dpi=300):
    """Convert PDF pages to images using PyMuPDF (no poppler needed)"""
    
    doc = fitz.open(pdf_path)
    output_paths = []
    
    pdf_file = Path(pdf_path)
    temp_dir = pdf_file.parent
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        mat = fitz.Matrix(dpi/72, dpi/72)  # Convert DPI to scale factor
        pix = page.get_pixmap(matrix=mat)
        
        image_path = temp_dir / f"{pdf_file.stem}_page_{page_num+1}.jpg"
        pix.save(str(image_path))
        output_paths.append(str(image_path))
    
    doc.close()
    return output_paths

async def process_invoice_with_retry(image_path, ws):
    from dotenv import load_dotenv
    load_dotenv()
    API_KEY = os.getenv('OPENAI_API_KEY')
    MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o')
    if not API_KEY:
        await ws.send_text("❌ OpenAI API key not found. Please check your .env file.")
        return None

    client = OpenAI(api_key=API_KEY)

    await ws.send_text("🔄 **Step 1:** Trying with original image...")
    result = await try_process_image(client, MODEL, image_path,ws,  is_preprocessed=False)
    condition = analyze_invoice_quality(result)

    if condition == 'not_invoice':
        await ws.send_text("❌ Uploaded file is not a valid invoice. Please upload a proper invoice document.")
        return None

    if condition == 'blur_too_bad':
        await ws.send_text("❌ Uploaded invoice is too blurry or unreadable.")
        return None

    if condition == 'blur_maybe':
        await ws.send_text("⚠️ Uploaded invoice is blurry; attempting enhancement.")
        preprocessed_path = preprocess_image_enhanced(image_path)
        download_preprocessed_image(preprocessed_path)
        result = await try_process_image(client, MODEL, preprocessed_path,ws, is_preprocessed=True)
        condition = analyze_invoice_quality(result)

        if condition == 'not_invoice':
            await ws.send_text("❌ Uploaded file is not a valid invoice after enhancement.")
            return None

        if condition == 'blur_too_bad':
            await ws.send_text("❌ Uploaded invoice is too blurry even after enhancement.")
            return None
    print("returning")
    return result
def analyze_invoice_quality(result):
    if not result:
        return 'no_data'
    qa = result.get('quality_assessment', {})
    can_extract = qa.get('can_extract_data', True)
    issues = qa.get('quality_issues', [])
    if not isinstance(issues, list):
        issues = []
    issues_lower = [str(i).strip().lower() for i in issues]
    qtp = qa.get('quality_too_poor', False)
    rs = qa.get('readability_score', '').lower()

    if not can_extract and 'not invoice' in issues_lower:
        return 'not_invoice'
    if not can_extract or qtp or rs == 'low':
        return 'blur_too_bad'
    if qtp or rs == 'medium':
        return 'blur_maybe'
    return 'good'

def preprocess_image_enhanced(image_path):
    """
    Enhanced preprocessing using OpenCV: grayscale, noise removal, adaptive binarization, morphological closing, and resize.
    Returns the processed image path (or original if failure).
    """
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found or invalid: {image_path}")
        
        # 1. Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 2. Noise reduction
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 3. Adaptive thresholding for binarization
        thresh = cv2.adaptiveThreshold(blur, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        
        # 4. Morphological closing
        kernel = np.ones((2, 2), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # 5. Resize if too large (like your Pillow logic)
        max_size = 2048
        height, width = morph.shape
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            morph = cv2.resize(morph, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)

        # Save processed image
        file_path = Path(image_path)
        processed_path = str(file_path.parent / f"{file_path.stem}_enhanced.jpg")
        cv2.imwrite(processed_path, morph)

        return processed_path
    
    except Exception as e:
        print(f"❌ OpenCV preprocessing failed: {e}")
        return image_path
    
def combine_pdf_page_results(page_results, pdf_filename):
    """Combine results from all PDF pages into a structured format"""
    
    successful_pages = [r for r in page_results if 'error' not in r]
    failed_pages = [r for r in page_results if 'error' in r]
    
    # Find the page with the most complete invoice header (usually page 1)
    main_page = None
    for page in successful_pages:
        if page.get('invoice_header', {}).get('vendor_name', 'N/A') != 'N/A':
            main_page = page
            break
    
    if not main_page and successful_pages:
        main_page = successful_pages[0]
    
    # Combine all line items from all pages
    all_line_items = []
    for page in successful_pages:
        page_line_items = page.get('line_items', [])
        for item in page_line_items:
            if item and any(v != 'N/A' and v != '' for v in item.values()):
                item['source_page'] = page.get('page_info', {}).get('page_number', 'Unknown')
                all_line_items.append(item)
    
    # 🔧 IMPROVED: Smart combination of terms and conditions from ALL pages
    combined_terms = {
        "payment_terms": "",
        "delivery_terms": "",
        "warranty_terms": "",
        "return_policy": "",
        "late_payment_charges": "",
        "jurisdiction": "",
        "other_conditions": []
    }
    
    # Collect the BEST (non-N/A) value for each term from all pages
    for page in successful_pages:
        page_terms = page.get('terms_and_conditions', {})
        page_num = page.get('page_info', {}).get('page_number', 'Unknown')
        
        for key, value in page_terms.items():
            if value and value != 'N/A' and ((isinstance(value, str) and value.strip()) or (isinstance(value, list) and len(value) > 0)):
                if key == 'other_conditions' and isinstance(value, list):
                    # Add list items
                    for condition in value:
                        if condition and condition != 'N/A':
                            combined_terms[key].append(condition)
                else:
                    # For single values, take the first meaningful one found
                    if not combined_terms[key] or combined_terms[key] == 'N/A':
                        combined_terms[key] = value
                    elif combined_terms[key] != value:
                        # If different values exist, combine them
                        combined_terms[key] = f"{combined_terms[key]} | {value}"
    
    # 🔧 IMPROVED: Smart combination of payment details from ALL pages
    combined_payment = {}
    
    # Start with main page payment details
    if main_page:
        combined_payment = main_page.get('payment_details', {}).copy()
    
    # Enhance with data from other pages
    for page in successful_pages:
        page_payment = page.get('payment_details', {})
        
        for key, value in page_payment.items():
            if value and value != 'N/A' and ((isinstance(value, str) and value.strip()) or (isinstance(value, list) and len(value) > 0)):

                if not combined_payment.get(key) or combined_payment.get(key) == 'N/A':
                    combined_payment[key] = value
                elif combined_payment[key] != value:
                    # If different values exist, combine them
                    combined_payment[key] = f"{combined_payment[key]} | {value}"
    
    # Create combined result
    combined_result = {
        'pdf_info': {
            'source_pdf': pdf_filename,
            'total_pages': len(page_results),
            'successful_pages': len(successful_pages),
            'failed_pages': len(failed_pages),
            'processing_date': datetime.now().isoformat()
        },
        'combined_data': {
            'invoice_header': main_page.get('invoice_header', {}) if main_page else {},
            'customer_details': main_page.get('customer_details', {}) if main_page else {},
            'line_items': all_line_items,
            'financial_summary': main_page.get('financial_summary', {}) if main_page else {},
            'payment_details': combined_payment,  # 🔧 Smart combined payment details
            'terms_and_conditions': combined_terms,  # 🔧 Smart combined terms
            'additional_info': main_page.get('additional_info', {}) if main_page else {}
        },
        'page_by_page_results': page_results,
        'processing_summary': {
            'total_line_items_found': len(all_line_items),
            'pages_with_line_items': len([p for p in successful_pages if p.get('line_items')]),
            'overall_quality': 'high' if len(successful_pages) == len(page_results) else 'medium' if successful_pages else 'low'
        }
    }
    
    return combined_result


def download_preprocessed_image(preprocessed_image_path):
    if not os.path.exists(preprocessed_image_path):
        return
    # with open(preprocessed_image_path, "rb") as file:
    #     st.download_button(
    #         label="📥 Download Preprocessed Image",
    #         data=file,
    #         file_name=os.path.basename(preprocessed_image_path),
    #         mime="image/jpeg"
    #     )

def save_result_to_file(result, filename, results_dir="resultjson"):
    """Save individual result to a separate JSON file"""
    if not result:
        return None
    
    # Create timestamp and unique ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    
    # Add metadata to result
    enhanced_result = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "processing_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "unique_id": unique_id,
            "filename": filename,
            "file_id": f"{timestamp}_{unique_id}"
        },
        "extraction_data": result
    }
    
    # Create individual file
    individual_filename = f"invoice_{timestamp}_{unique_id}.json"
    individual_path = os.path.join(results_dir, individual_filename)
    
    with open(individual_path, 'w', encoding='utf-8') as f:
        json.dump(enhanced_result, f, indent=2, ensure_ascii=False)
    
    return individual_path, enhanced_result

def append_to_master_results(result, filename, results_dir="resultjson"):
    """Append result to master results file"""
    master_file = os.path.join(results_dir, "all_results.json")
    
    # Create timestamp and unique ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    
    # Prepare new entry
    new_entry = {
        "id": f"{timestamp}_{unique_id}",
        "timestamp": datetime.now().isoformat(),
        "processing_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_filename": filename,
        "extraction_data": result
    }
    
    # Load existing results or create new list
    if os.path.exists(master_file):
        try:
            with open(master_file, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
            if not isinstance(all_results, list):
                all_results = [all_results]  # Convert single object to list
        except (json.JSONDecodeError, FileNotFoundError):
            all_results = []
    else:
        all_results = []
    
    # Append new result
    all_results.append(new_entry)
    
    # Save back to master file
    with open(master_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    return master_file, len(all_results)