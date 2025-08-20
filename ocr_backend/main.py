
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import json
from process import process_pdf, process_image

# New imports for assessment
import os
from dotenv import load_dotenv
from openai import OpenAI
from decimal import Decimal
from datetime import datetime
import sys
import importlib.util
from pathlib import Path

# Dynamic loader for financial validation modules (supports external placement)
init_sqlite_db = None
FinancialLogicAgent = None
User = None
Role = None
LineItem = None
ExpenseReport = None
ExpenseCategory = None
seed_sample_data = None

def _try_load_module_from_path(module_name: str, file_path: str):
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    except Exception:
        return None
    return None

def _ensure_financial_modules() -> bool:
    global init_sqlite_db, FinancialLogicAgent, User, Role, LineItem, ExpenseReport, seed_sample_data
    # 1) Try normal import if placed in PYTHONPATH
    try:
        from financial_logic_agent import (
            init_sqlite_db as _init,
            FinancialLogicAgent as _Agent,
            User as _User,
            Role as _Role,
            LineItem as _LineItem,
            ExpenseReport as _Report,
            ExpenseCategory as _Category,
        )
        init_sqlite_db = _init
        FinancialLogicAgent = _Agent
        User = _User
        Role = _Role
        LineItem = _LineItem
        ExpenseReport = _Report
        ExpenseCategory = _Category
    except Exception:
        pass

    try:
        from db import insert_sample_data as _seed
        seed_sample_data = _seed
    except Exception:
        pass

    if FinancialLogicAgent is not None:
        return True

    # 2) Try to locate files on disk (parent directories, Downloads, ENV var)
    candidates = []
    env_dir = os.getenv('FIN_AGENT_PATH')
    if env_dir:
        candidates.append(Path(env_dir))

    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        candidates.append(p)

    # common Windows Downloads folder
    candidates.append(Path.home() / 'Downloads')

    fin_mod = None
    db_mod = None
    for base in candidates:
        fin_path = base / 'financial_logic_agent.py'
        if fin_path.is_file():
            fin_mod = _try_load_module_from_path('financial_logic_agent', str(fin_path))
        db_path = base / 'db.py'
        if db_path.is_file():
            db_mod = _try_load_module_from_path('db', str(db_path))
        if fin_mod and db_mod:
            break

    if fin_mod:
        init_sqlite_db = getattr(fin_mod, 'init_sqlite_db', None)
        FinancialLogicAgent = getattr(fin_mod, 'FinancialLogicAgent', None)
        User = getattr(fin_mod, 'User', None)
        Role = getattr(fin_mod, 'Role', None)
        LineItem = getattr(fin_mod, 'LineItem', None)
        ExpenseReport = getattr(fin_mod, 'ExpenseReport', None)
        ExpenseCategory = getattr(fin_mod, 'ExpenseCategory', None)

    if db_mod:
        seed_sample_data = getattr(db_mod, 'insert_sample_data', None)

    return (
        FinancialLogicAgent is not None and User is not None and Role is not None and
        LineItem is not None and ExpenseReport is not None and ExpenseCategory is not None
    )

app = FastAPI()

# Enable CORS for all origins (for both REST API and WebSockets)
origins = [
    "http://localhost:5173",  # Your React app's frontend URL
    "http://localhost:3000",  # If you use another port, you can add that too
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows CORS from these origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allow all headers
)

# WebSocket endpoint to process the file
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        json_file_path = "invoice_result.json"
        with open(json_file_path, "r", encoding="utf-8") as f:
            file_data = json.load(f)
        # Receive the file from the frontend (simulated by the uploaded file path here)
        file_name = await websocket.receive_text() 
        uploaded_file = await websocket.receive_bytes()  # Get file in bytes
        

        # # Save the uploaded file
        #  # Generate a unique file name
        # file_path = os.path.join("uploads", file_name)
        # os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # with open(file_path, "wb") as f:
        #     f.write(uploaded_file)

        # Send the file format
        file_extension = file_name.split('.')[-1].lower()
        await websocket.send_text(json.dumps({"message" : f"File format: {file_extension.upper()}", "type" : "success", "finished" : True} ))

        # Send the file size
        # file_size = os.path.getsize(file_path)
        # await websocket.send_text(f"File size: {file_size} bytes")

        if file_extension == "pdf":
            await websocket.send_text("📄 **PDF file detected**")
            # await websocket.send_json({"result" : {"text" : file_data} })
            processing_result = await process_pdf(uploaded_file, websocket, file_name)
            
        else:
            #  await process_image(uploaded_file, websocket, file_name)
            await websocket.send_text("📄 **Image detected**")
            await process_image(uploaded_file, websocket, file_name)     
            # await websocket.send_json({"result" : {"text" : file_data} })
        # Simulate processing completion
        await websocket.send_text(json.dumps({"message" : "File uploaded and processed successfully!", "type" : "success"}))
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(e)

# New REST endpoint: Assess whether purchase request and OCR-extracted invoice match
@app.post("/assess")
async def assess_invoice_match(payload: dict = Body(...)):
    """
    Expects JSON body with:
    {
      "purchase": { ... purchase request fields incl. items ... },
      "invoice": { ... OCR extracted JSON ... }
    }
    Returns structured assessment with overall_match, score, discrepancies, and recommendation.
    """
    try:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        if not api_key:
            return JSONResponse(status_code=500, content={"message": "OPENAI_API_KEY not configured"})

        client = OpenAI(api_key=api_key)

        purchase = payload.get("purchase", {})
        invoice = payload.get("invoice", {})

        system_prompt = (
            "You are a strict auditor comparing a purchase/expense request against an OCR-extracted supplier invoice. "
            "Return ONLY valid JSON with fields: overall_match (boolean), score (0-100), summary (string), "
            "discrepancies (array of {field, expected, found, note}), and recommendation ('approve' or 'reject'). "
            "Vendor name MUST match strictly (case-insensitive exact match ignoring minor punctuation/spacing). If vendor differs, this is a critical discrepancy. "
            "IMPORTANT TOTALS POLICY: Compare ONLY the purchase request 'amount' (pre-tax total) to the invoice pre-tax amount 'financial_summary.subtotal'. "
            "Do NOT compare against invoice grand total and do NOT penalize tax differences. If the invoice has no explicit subtotal, SKIP the amount comparison altogether "
            "(do not reject solely due to missing subtotal). Treat deltas ≤ 1% as minor and acceptable. "
            "FUZZY ITEM MATCHING POLICY: Item names can differ (e.g., 'blue pen' vs 'cello blue ball pen'). Use fuzzy/approximate matching – token-based, case-insensitive. "
            "BRAND/STOPWORD HANDLING: Ignore vendor/brand tokens and corporate suffixes when comparing item names. Remove tokens appearing in vendorName, and common brand/corporate terms (e.g., 'cello', 'pvt', 'ltd', 'inc', 'co', 'company'). "
            "BIDIRECTIONAL OVERLAP: Count as a name match when key tokens substantially overlap in EITHER direction (purchase↔invoice); aim for token overlap/Jaccard ≥ ~0.4 once brand tokens are removed. "
            "SYNONYMS: Treat domain synonyms as equivalent (e.g., 'pen' ~ 'ball pen' ~ 'ballpoint pen' ~ 'gel pen'). "
            "Use quantity and unit/line price proximity (≤ 2% delta) to reinforce mapping. Prefer one-to-one mappings maximizing similarity and price proximity. "
            "Do NOT flag cosmetic naming differences when quantities and prices align within tolerance."
        )

        user_prompt = (
            "Compare these two JSON documents. REQUIRE vendorName to match (case-insensitive; ignore punctuation/spacing). "
            "Fuzzy item matching: remove vendor/brand tokens from both item names (including tokens from vendorName and common corporate suffixes), then perform bidirectional token-overlap matching. "
            "Consider domain synonyms (e.g., 'pen' ~ 'ball pen' ~ 'ballpoint pen' ~ 'gel pen'). Quantities and unit/line prices must be within tolerance. "
            "Amount comparison: ONLY compare purchase amount vs invoice financial_summary.subtotal. If subtotal is missing, skip the amount comparison and do not penalize.\n\n"
            f"PURCHASE_REQUEST_JSON:\n{json.dumps(purchase, ensure_ascii=False)}\n\n"
            f"INVOICE_OCR_JSON:\n{json.dumps(invoice, ensure_ascii=False)}\n\n"
            "Respond with JSON only: {\n"
            "  \"overall_match\": true/false,\n"
            "  \"score\": number,\n"
            "  \"summary\": \"...\",\n"
            "  \"discrepancies\": [ { \"field\": \"...\", \"expected\": \"...\", \"found\": \"...\", \"note\": \"...\" } ],\n"
            "  \"recommendation\": \"approve\"|\"reject\"\n"
            "}"
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=900
        )

        content = response.choices[0].message.content.strip()
        # Try to load as JSON directly; if model wrapped in code fences, strip them
        if content.startswith('```'):
            content = content.split('\n', 1)[1]
            if content.endswith('```'):
                content = content[:-3]
        assessment = json.loads(content)
        return assessment
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

# Helper to extract line items from OCR JSON with best-effort mapping
def _coalesce(*values):
    for v in values:
        if v is not None:
            return v
    return None

def _to_decimal_safe(value, default: str = "0"):
    try:
        if value is None:
            return Decimal(default)
        return Decimal(str(value))
    except Exception:
        return Decimal(default)

def _extract_invoice_core(invoice: dict) -> dict:
    # Support multiple shapes: wrapper with extraction_data, combined_data, or raw
    if not isinstance(invoice, dict):
        return {}
    if 'extraction_data' in invoice and isinstance(invoice['extraction_data'], dict):
        return invoice['extraction_data']
    if 'combined_data' in invoice and isinstance(invoice['combined_data'], dict):
        return invoice['combined_data']
    return invoice

def _build_report_from_purchase_and_invoice(purchase: dict, invoice: dict) -> ExpenseReport:
    core = _extract_invoice_core(invoice)
    header = core.get('invoice_header', {}) if isinstance(core, dict) else {}
    fin = core.get('financial_summary', {}) if isinstance(core, dict) else {}
    items = core.get('line_items', []) if isinstance(core, dict) else []

    vendor_name = _coalesce(purchase.get('vendorName'), header.get('vendor_name'), '')
    currency = _coalesce(purchase.get('currency'), fin.get('currency'), 'USD')
    jurisdiction = purchase.get('jurisdiction') or 'US'

    # Dates
    inv_date_raw = _coalesce(header.get('invoice_date'), header.get('date'))
    try:
        expense_date = datetime.fromisoformat(str(inv_date_raw)) if inv_date_raw else datetime.utcnow()
    except Exception:
        expense_date = datetime.utcnow()
    submission_date = datetime.utcnow()

    # Build line items from OCR
    parsed_items = []
    if isinstance(items, list):
        for it in items:
            if not isinstance(it, dict):
                continue
            desc = _coalesce(it.get('description'), it.get('item'), it.get('name'), '') or ''
            qty = _to_decimal_safe(_coalesce(it.get('quantity'), it.get('qty'), 1), '1')
            unit = _to_decimal_safe(_coalesce(it.get('unit_price'), it.get('unit price'), it.get('rate'), 0), '0')
            # tax rate may be a number like 0.08 or 8 -> normalize if > 1
            tr = _to_decimal_safe(_coalesce(it.get('tax_rate'), it.get('tax'), 0), '0')
            tax_rate = (tr / Decimal('100')) if tr > 1 else tr
            item_currency = it.get('currency') or currency
            # Guard if ExpenseCategory not loaded for any reason
            cat_default = ExpenseCategory.UNKNOWN if ExpenseCategory is not None else 'unknown'
            try:
                line = LineItem(
                    description=str(desc),
                    quantity=qty,
                    unit_price=unit,
                    tax_rate=tax_rate,
                    category=cat_default,
                    currency=item_currency,
                    jurisdiction=jurisdiction,
                    vendor=vendor_name
                )
            except Exception:
                # Some versions may require string category; fall back
                line = LineItem(
                    description=str(desc),
                    quantity=qty,
                    unit_price=unit,
                    tax_rate=tax_rate,
                    category=cat_default,
                    currency=item_currency,
                    jurisdiction=jurisdiction,
                    vendor=vendor_name
                )
            parsed_items.append(line)

    # Fallback: if no OCR items, synthesize from purchase items
    if not parsed_items and isinstance(purchase.get('items'), list):
        for it in purchase['items']:
            if not isinstance(it, dict):
                continue
            desc = _coalesce(it.get('description'), it.get('title'), '')
            qty = _to_decimal_safe(_coalesce(it.get('quantity'), 1), '1')
            unit = _to_decimal_safe(_coalesce(it.get('unitPrice'), it.get('unit_price'), 0), '0')
            cat_default = ExpenseCategory.UNKNOWN if ExpenseCategory is not None else 'unknown'
            try:
                line = LineItem(
                    description=str(desc),
                    quantity=qty,
                    unit_price=unit,
                    tax_rate=Decimal('0'),
                    category=cat_default,
                    currency=currency,
                    jurisdiction=jurisdiction,
                    vendor=vendor_name
                )
            except Exception:
                line = LineItem(
                    description=str(desc),
                    quantity=qty,
                    unit_price=unit,
                    tax_rate=Decimal('0'),
                    category=cat_default,
                    currency=currency,
                    jurisdiction=jurisdiction,
                    vendor=vendor_name
                )
            parsed_items.append(line)

    claimed_total = _to_decimal_safe(_coalesce(fin.get('total_amount'), purchase.get('amount'), 0), '0')

    # Minimal user and report
    employee_user = User(user_id=purchase.get('employeeId') or 'employee', role=Role.EMPLOYEE)
    report = ExpenseReport(
        report_id=purchase.get('title') or vendor_name or 'report',
        employee=employee_user,
        submission_date=submission_date,
        expense_date=expense_date,
        line_items=parsed_items,
        currency=currency,
        jurisdiction=jurisdiction,
        claimed_total=claimed_total,
        invoice_metadata={
            'subtotal': fin.get('subtotal'),
            'total_tax_amount': fin.get('total_tax_amount') or fin.get('tax_total'),
            'total_amount': fin.get('total_amount') or fin.get('total')
        }
    )
    # Sanitize categories: ensure enums, not strings
    def _coerce_category_enum(cat):
        if ExpenseCategory is None:
            return cat
        if cat is None:
            return ExpenseCategory.UNKNOWN
        if isinstance(cat, str):
            try:
                return ExpenseCategory(cat.strip().lower())
            except Exception:
                return ExpenseCategory.UNKNOWN
        return cat
    for it in report.line_items:
        try:
            it.category = _coerce_category_enum(it.category)
        except Exception:
            try:
                it.category = ExpenseCategory.UNKNOWN if ExpenseCategory is not None else it.category
            except Exception:
                pass
    return report

def _sanitize_report_categories(report: ExpenseReport):
    if not report or not getattr(report, 'line_items', None):
        return
    for it in report.line_items:
        try:
            if isinstance(getattr(it, 'category', None), str):
                if ExpenseCategory is not None:
                    try:
                        it.category = ExpenseCategory(it.category.strip().lower())
                    except Exception:
                        it.category = ExpenseCategory.UNKNOWN
            elif it.category is None and ExpenseCategory is not None:
                it.category = ExpenseCategory.UNKNOWN
        except Exception:
            try:
                if ExpenseCategory is not None:
                    it.category = ExpenseCategory.UNKNOWN
            except Exception:
                pass

@app.post("/financial-validate")
async def financial_validate(payload: dict = Body(...)):
    try:
        load_dotenv()
        # Ensure ALL required symbols are present (not just one)
        if any(x is None for x in [FinancialLogicAgent, User, Role, LineItem, ExpenseReport, ExpenseCategory]):
            loaded = _ensure_financial_modules()
            if not loaded:
                return JSONResponse(status_code=500, content={"message": "Financial validation module not available"})

        # Initialize DB once per process (idempotent)
        try:
            if init_sqlite_db:
                await init_sqlite_db()
            if seed_sample_data:
                try:
                    seed_sample_data()
                except Exception:
                    pass
        except Exception:
            pass

        purchase = payload.get('purchase', {})
        invoice = payload.get('invoice', {})

        report = _build_report_from_purchase_and_invoice(purchase, invoice)
        # Final safety: coerce any string categories to enums
        _sanitize_report_categories(report)
        agent = FinancialLogicAgent(report.employee)
        result = await agent.validate_expense_report(report)

        return {
            'is_valid': bool(result.is_valid),
            'violations': result.violations,
            'warnings': result.warnings,
            'risk_score': float(result.risk_score),
            'recommended_action': result.recommended_action
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})
