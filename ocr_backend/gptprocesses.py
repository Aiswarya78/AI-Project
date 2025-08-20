import base64
import json

async def try_process_image(client, model, image_path, ws, is_preprocessed=False):
    """Single attempt to process image with GPT-4o with enhanced error handling"""
    base64_image = encode_image(image_path)
    
    # Enhanced prompt that explicitly asks about quality issues
    prompt = f"""
STRICT INSTRUCTION: Only output valid JSON, no markdown or explanations.

{'RETRY ATTEMPT - This is a preprocessed image.' if is_preprocessed else 'FIRST ATTEMPT - This is the original image.'}

First, assess if you can reliably extract data from this image:
- If the image is too blurry, dark, or distorted to read text clearly, set "quality_too_poor" to true
- If you can read most text despite some quality issues, set "quality_too_poor" to false
- If this image is not a valid invoice, set "can_extract_data" to false and add "not invoice" to "quality_issues".


Extract ALL available information from this invoice and return as JSON:
{{
  "quality_assessment": {{
    "quality_too_poor": true/false,
    "quality_issues": ["list any specific quality problems"],
    "readability_score": "high/medium/low",
    "can_extract_data": true/false,
    "preprocessing_recommended": true/false
  }},
  "invoice_header": {{
    "vendor_name": "",
    "vendor_address": "",
    "vendor_phone": "",
    "vendor_email": "",
    "vendor_website": "",
    "vendor_gst_number": "",
    "vendor_pan": "",
    "invoice_number": "",
    "invoice_date": "",
    "due_date": "",
    "purchase_order_number": "",
    "reference_number": "",
    "currency": ""
  }},
  "customer_details": {{
    "customer_name": "",
    "customer_address": "",
    "customer_phone": "",
    "customer_email": "",
    "customer_gst_number": "",
    "customer_pan": "",
    "billing_address": "",
    "shipping_address": "",
    "customer_contact_person": ""
  }},
  "line_items": [
    {{
      "item_number": "",
      "description": "",
      "hsn_sac_code": "",
      "quantity": "",
      "unit": "",
      "unit_price": "",
      "discount": "",
      "tax_rate": "",
      "tax_amount": "",
      "total_price": ""
    }}
  ],
  "financial_summary": {{
    "subtotal": "",
    "total_discount": "",
    "taxable_amount": "",
    "cgst": "",
    "sgst": "",
    "igst": "",
    "cess": "",
    "other_charges": "",
    "shipping_charges": "",
    "total_tax_amount": "",
    "round_off": "",
    "total_amount": "",
    "amount_in_words": ""
  }},
  "payment_details": {{
    "payment_terms": "",
    "payment_method": "",
    "bank_name": "",
    "account_number": "",
    "ifsc_code": "",
    "branch": "",
    "upi_id": "",
    "advance_paid": "",
    "balance_due": ""
  }},
  "terms_and_conditions": {{
    "payment_terms": "",
    "delivery_terms": "",
    "warranty_terms": "",
    "return_policy": "",
    "late_payment_charges": "",
    "jurisdiction": "",
    "other_conditions": []
  }},
  "additional_info": {{
    "notes": "",
    "special_instructions": "",
    "delivery_date": "",
    "place_of_supply": "",
    "reverse_charge": "",
    "document_type": "",
    "series": "",
    "authorised_signatory": "",
    "stamp_or_seal": "",
    "qr_code_present": ""
  }},
  "detection_metadata": {{
    "tables_detected": true/false,
    "handwriting_detected": true/false,
    "logo_detected": true/false,
    "stamp_detected": true/false,
    "signature_detected": true/false,
    "barcode_qr_detected": true/false,
    "multi_page_document": true/false,
    "document_quality": "high/medium/low",
    "extraction_confidence": "high/medium/low",
    "unclear_fields": []
  }}
}}

INSTRUCTIONS:
- Be honest about image quality in the quality_assessment section
- If quality_too_poor is true, still try to extract what you can see
- For missing/unclear fields, use "N/A"
# In your try_process_image function, enhance the currency instructions:

CURRENCY DETECTION - IMPORTANT:
- ALWAYS preserve currency symbols in amounts: $154.06, ₹10,000, €500, etc.
- Include currency symbols in ALL amount fields: total_amount, subtotal, unit_price, etc.
- Do NOT extract just numbers - include the currency symbol with the number
- Look for currency symbols: ₹, $, €, £, ¥, etc.
- Look for currency codes: INR, USD, EUR, GBP, etc.
- Look for currency words: Rupees, Dollars, Euros, Pounds, etc.
- Extract currency in BOTH invoice_header and financial_summary sections
- If amounts have symbols like $100.00, preserve the $ in the JSON output

- If text is completely unreadable due to quality, mention this in quality_issues
- Extract ALL visible text and data fields
- For terms and conditions, extract the full text even if lengthy
- Include any fine print, disclaimers, or legal text
- Capture payment terms like "Net 30", "Due on receipt", etc.
- Extract tax breakdowns (CGST, SGST, IGST) if present
- Include any special notes, delivery instructions, or remarks
- Identify HSN/SAC codes for items if visible
- Extract complete addresses with pin codes
- Include contact details like phone, email, website
- Capture bank details for payments
- Note any stamps, signatures, or authentication marks
- Return only valid JSON without any explanation.
"""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=4000,  # 🔧 INCREASED from 2500 to 4000
            temperature=0.1
        )
        
        content = response.choices[0].message.content
        cleaned_content = clean_json_response(content)
        result = json.loads(cleaned_content)

# 🔧 NEW: Enhance currency detection
        result = enhance_currency_detection(result)

        return result

        
    except json.JSONDecodeError as e:
        ws.send_text(f"❌ JSON parsing error: {e}")
        # 🔧 NEW: Try simplified extraction for large documents
        return try_simplified_extraction(client, model, image_path, is_preprocessed)
    except Exception as e:
        ws.send_text(f"❌ Processing error: {e}")
        return None


def encode_image(file_path):
    """Encode image to base64"""
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
def clean_json_response(content):
    """Clean markdown formatting and fix truncated JSON"""
    content = content.strip()
    
    # Remove code block markers
    if content.startswith('```json'):
        content = content[7:]  # Remove ```
    elif content.startswith('```'):
        content = content[3:]   # Remove ```
    
    if content.endswith('```'):
        content = content[:-3]  # Remove trailing ```
    
    content = content.strip()
    
    # 🔧 NEW: Fix common JSON truncation issues for large documents
    if not content.endswith('}'):
        # Remove trailing commas and incomplete entries
        content = content.rstrip(',\n\r\t ')
        
        # Try to close incomplete strings
        if content.count('"') % 2 == 1:
            content += '"'
        
        # Close incomplete objects and arrays
        open_braces = content.count('{') - content.count('}')
        open_brackets = content.count('[') - content.count(']')
        
        if open_brackets > 0:
            content += ']' * open_brackets
        if open_braces > 0:
            content += '}' * open_braces
    
    return content

def enhance_currency_detection(result):
    """Enhance currency detection and set defaults"""
    if not result:
        return result
    
    # Currency patterns to look for
    currency_patterns = {
        '₹': 'INR', 'Rs.': 'INR', 'Rs': 'INR', 'INR': 'INR', 'Rupees': 'INR', 'rupees': 'INR',
        '$': 'USD', 'USD': 'USD', 'Dollars': 'USD',
        '€': 'EUR', 'EUR': 'EUR', 'Euros': 'EUR',
        '£': 'GBP', 'GBP': 'GBP', 'Pounds': 'GBP'
    }
    
    detected_currency = None
    
    # 🔧 PRIORITY 1: Look for symbols in amounts FIRST (most reliable)
    amounts_to_check = []
    
    if result.get('financial_summary'):
        amounts_to_check.extend([
            result['financial_summary'].get('total_amount', ''),
            result['financial_summary'].get('subtotal', ''),
            result['financial_summary'].get('amount_in_words', '')
        ])
    
    if result.get('line_items'):
        for item in result['line_items']:
            amounts_to_check.extend([item.get('unit_price', ''), item.get('total_price', '')])
    
    # Look for currency patterns in amounts
    for amount in amounts_to_check:
        if amount and amount != 'N/A':
            for pattern, currency_code in currency_patterns.items():
                if pattern in str(amount):
                    detected_currency = currency_code
                    print(f"DEBUG: Found '{pattern}' in amount '{amount}' -> Currency: {currency_code}")
                    break
            if detected_currency:
                break
    
    # 🔧 PRIORITY 2: Check explicit currency fields (fallback)
    if not detected_currency:
        header_currency = result.get('invoice_header', {}).get('currency', 'N/A')
        financial_currency = result.get('financial_summary', {}).get('currency', 'N/A')
        
        if header_currency and header_currency != 'N/A':
            detected_currency = header_currency
        elif financial_currency and financial_currency != 'N/A':
            detected_currency = financial_currency
    
    # 🔧 PRIORITY 3: Check for Indian tax indicators (secondary fallback)
    if not detected_currency or detected_currency == 'N/A':
        has_indian_tax = False
        
        # Check for Indian indicators
        if result.get('financial_summary'):
            financial = result['financial_summary']
            if (financial.get('cgst', 'N/A') != 'N/A' or 
                financial.get('sgst', 'N/A') != 'N/A' or 
                financial.get('igst', 'N/A') != 'N/A'):
                has_indian_tax = True
        
        if result.get('invoice_header'):
            header = result['invoice_header']
            if (header.get('vendor_gst_number', 'N/A') != 'N/A' or 
                header.get('vendor_pan', 'N/A') != 'N/A'):
                has_indian_tax = True
        
        # Only default to INR if Indian indicators found
        if has_indian_tax:
            detected_currency = 'INR'
        else:
            # If no Indian indicators and no symbols found, this is suspicious
            print("DEBUG: No currency symbols or Indian indicators found - defaulting to INR")
            detected_currency = 'INR'
    
    # Normalize currency
    currency_map = {
        'INR': 'Rupees (₹)', 'Rupees': 'Rupees (₹)', 'rupees': 'Rupees (₹)', 
        'Rs': 'Rupees (₹)', 'Rs.': 'Rupees (₹)',
        'USD': 'US Dollars ($)', 'Dollars': 'US Dollars ($)',
        'EUR': 'Euros (€)', 'GBP': 'British Pounds (£)'
    }
    
    final_currency = currency_map.get(detected_currency, 'Rupees (₹)')
    
    # Update result with detected currency
    if 'invoice_header' in result:
        result['invoice_header']['currency'] = final_currency
    if 'financial_summary' in result:
        result['financial_summary']['currency'] = final_currency
    
    print(f"DEBUG: Final currency set to: {final_currency}")
    
    return result

def try_simplified_extraction(client, model, image_path, is_preprocessed=False):
    """Simplified extraction for large documents that cause JSON truncation"""
    base64_image = encode_image(image_path)
    
    # Simplified prompt focusing on key data
    prompt = f"""
STRICT INSTRUCTION: Only output valid JSON, no markdown or explanations.

This is a large invoice. Extract key information and SUMMARIZE line items instead of listing all individually:

{{
  "quality_assessment": {{
    "quality_too_poor": false,
    "quality_issues": ["Large document - simplified extraction"],
    "readability_score": "high",
    "can_extract_data": true,
    "preprocessing_recommended": false
  }},
  "invoice_header": {{
    "vendor_name": "",
    "vendor_address": "",
    "invoice_number": "",
    "invoice_date": "",
    "total_amount": ""
  }},
  "customer_details": {{
    "customer_name": "",
    "customer_address": ""
  }},
  "line_items_summary": {{
    "total_line_items": 0,
    "sample_items": ["List first 3-5 items only"],
    "total_value": "",
    "note": "Large document - showing sample items only"
  }},
  "financial_summary": {{
    "total_amount": "",
    "currency": ""
  }},
  "terms_and_conditions": {{
    "payment_terms": "",
    "other_conditions": []
  }}
}}

Extract main invoice details and count/sample the line items instead of listing all items individually.
"""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1500,  # Lower token limit for simplified response
            temperature=0.1
        )
        
        content = response.choices.message.content
        cleaned_content = clean_json_response(content)
        result = json.loads(cleaned_content)
        
        # Convert simplified result to standard format
        return convert_simplified_to_standard_format(result)
        
    except Exception as e:
        print("❌ Simplified extraction also failed: {e}")
        return None
def convert_simplified_to_standard_format(simplified_result):
    """Convert simplified extraction result to standard format"""
    standard_result = {
        "quality_assessment": simplified_result.get("quality_assessment", {}),
        "invoice_header": simplified_result.get("invoice_header", {}),
        "customer_details": simplified_result.get("customer_details", {}),
        "line_items": [],  # Empty for large documents
        "financial_summary": simplified_result.get("financial_summary", {}),
        "payment_details": {},
        "terms_and_conditions": simplified_result.get("terms_and_conditions", {}),
        "additional_info": {
            "notes": f"Large document with {simplified_result.get('line_items_summary', {}).get('total_line_items', 'many')} items - simplified extraction used"
        },
        "detection_metadata": {
            "large_document": True,
            "extraction_method": "simplified"
        }
    }
    
    # Add sample items if available
    sample_items = simplified_result.get("line_items_summary", {}).get("sample_items", [])
    for i, item_desc in enumerate(sample_items[:5], 1):
        standard_result["line_items"].append({
            "item_number": f"Sample {i}",
            "description": item_desc,
            "quantity": "N/A",
            "unit_price": "N/A",
            "total_price": "N/A",
            "note": "Sample item from large document"
        })
    
    return standard_result