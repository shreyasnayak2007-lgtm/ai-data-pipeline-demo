"""AI data pipeline demo: messy retail data -> cleaned, AI-categorized, dashboard-ready tables.

Steps: load -> clean -> categorize (LLM) -> aggregate -> push to Google Sheets (or CSV).
"""
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

RAW_FILE = Path(os.getenv("RAW_FILE", "data/Online Retail.xlsx"))
CACHE_FILE = Path("category_cache.json")
OUT_DIR = Path("output")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
BATCH_SIZE = 100  # Increased for higher token efficiency
MAX_WORKERS = 5   # Parallel API call threads
CATEGORIES = [
    "Home Decor", "Kitchen & Dining", "Gifts & Novelties", "Stationery & Cards",
    "Lighting & Candles", "Garden & Outdoor", "Toys & Games",
    "Bags & Accessories", "Party & Seasonal", "Storage & Organization", "Other",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pipeline")


# 1. LOAD ---------------------------------------------------------------------
def load() -> pd.DataFrame:
    log.info("Loading %s", RAW_FILE)
    if RAW_FILE.suffix.lower() == ".csv":
        df = pd.read_csv(RAW_FILE, encoding="ISO-8859-1", engine="c")
    else:
        df = pd.read_excel(RAW_FILE)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    log.info("Loaded %d rows", len(df))
    return df


# 2. CLEAN --------------------------------------------------------------------
def clean(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    # Filter valid dates and descriptions in one pass
    df = df.dropna(subset=["Description", "InvoiceDate"])

    # Vectorized string cleaning
    df["Description"] = (
        df["Description"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True).str.title()
    )
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    
    # Fast boolean indexing
    cancellations = df["InvoiceNo"].str.startswith("C")
    missing_cust = df["CustomerID"].isna()
    
    # Filter positive prices only
    df = df[df["UnitPrice"] > 0].copy()
    
    df["IsCancellation"] = cancellations
    df["MissingCustomerID"] = missing_cust
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    df["Date"] = df["InvoiceDate"].dt.date
    df["Month"] = df["InvoiceDate"].dt.to_period("M").astype(str)
    
    log.info("Clean: %d -> %d rows (%d cancellations, %d missing CustomerID)",
             n0, len(df), df["IsCancellation"].sum(), df["MissingCustomerID"].sum())
    return df


# 3. CATEGORIZE (PARALLEL & ASYNC BATCHING) -----------------------------------
def _ask_llm(client: OpenAI, items: list[str]) -> dict:
    prompt = (
        "Categorize each retail product description into exactly one category from this list: "
        f"{CATEGORIES}.\nReturn JSON of the form {{\"results\": {{\"<description>\": \"<category>\"}}}} "
        "using the descriptions exactly as given.\n\nDescriptions:\n" + json.dumps(items)
    )
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
            )
            return json.loads(resp.choices[0].message.content).get("results", {})
        except Exception as exc:
            wait = 2 ** attempt * 2
            log.warning("LLM call failed (%s); retry in %ss", exc, wait)
            time.sleep(wait)
    log.error("Batch failed after retries; marking items as Other")
    return {}


def categorize(df: pd.DataFrame) -> pd.DataFrame:
    # Load cache if valid
    cache = {}
    if CACHE_FILE.exists() and CACHE_FILE.stat().st_size > 0:
        try:
            cache = json.loads(CACHE_FILE.read_text())
        except Exception:
            cache = {}

    todo = [d for d in df["Description"].unique() if d not in cache]
    log.info("%d unique descriptions, %d need categorizing", df["Description"].nunique(), len(todo))

    if todo:
        client = OpenAI()
        batches = [todo[i:i + BATCH_SIZE] for i in range(0, len(todo), BATCH_SIZE)]
        
        # Process batches concurrently
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_batch = {executor.submit(_ask_llm, client, batch): batch for batch in batches}
            
            completed = 0
            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                result = future.result()
                for d in batch:
                    cat = result.get(d, "Other")
                    cache[d] = cat if cat in CATEGORIES else "Other"
                
                completed += len(batch)
                log.info("Categorized %d/%d", completed, len(todo))

        # Atomic flush to disk
        CACHE_FILE.write_text(json.dumps(cache, indent=1))

    df["Category"] = df["Description"].map(cache).fillna("Other")
    return df


# 4. AGGREGATE ----------------------------------------------------------------
def aggregate(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    sales = df[~df["IsCancellation"]]
    returns = df[df["IsCancellation"]]

    daily = sales.groupby("Date").agg(
        Revenue=("Revenue", "sum"), Orders=("InvoiceNo", "nunique")).reset_index()

    by_category = sales.groupby("Category").agg(
        Revenue=("Revenue", "sum"), Units=("Quantity", "sum")).reset_index() \
        .sort_values("Revenue", ascending=False)

    top_products = sales.groupby(["Description", "Category"]).agg(
        Revenue=("Revenue", "sum"), Units=("Quantity", "sum")).reset_index() \
        .sort_values("Revenue", ascending=False).head(10)

    monthly_sales = sales.groupby("Month")["Revenue"].sum()
    monthly_returns = returns.groupby("Month")["Revenue"].sum().abs()
    return_rate = pd.DataFrame({"Sales": monthly_sales, "Returns": monthly_returns}).fillna(0)
    return_rate["ReturnRate"] = (return_rate["Returns"] / return_rate["Sales"]).round(4)
    return_rate = return_rate.reset_index()

    kpis = pd.DataFrame([{
        "TotalRevenue": round(sales["Revenue"].sum(), 2),
        "Orders": sales["InvoiceNo"].nunique(),
        "UniqueCustomers": int(sales["CustomerID"].nunique()),
        "ReturnRate": round(returns["Revenue"].abs().sum() / sales["Revenue"].sum(), 4) if sales["Revenue"].sum() > 0 else 0,
    }])

    return {"kpis": kpis, "daily_revenue": daily, "revenue_by_category": by_category,
            "top_products": top_products, "return_rate_monthly": return_rate}


# 5. PUSH ---------------------------------------------------------------------
def push_to_sheets(tables: dict[str, pd.DataFrame]) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    for name, t in tables.items():
        t.to_csv(OUT_DIR / f"{name}.csv", index=False)
    log.info("Wrote CSVs to %s/", OUT_DIR)

    sheet_id = os.getenv("GOOGLE_SHEET_ID")
    creds = os.getenv("GOOGLE_CREDS_FILE", "service_account.json")
    if not sheet_id or not Path(creds).exists():
        log.warning("No Sheets config found; skipping Sheets push (upload CSVs manually)")
        return
    import gspread
    sh = gspread.service_account(filename=creds).open_by_key(sheet_id)
    for name, t in tables.items():
        t_str = t.astype(str)
        try:
            ws = sh.worksheet(name)
            ws.clear()
        except gspread.WorksheetNotFound:
            ws = sh.add_worksheet(title=name, rows=len(t_str) + 5, cols=len(t_str.columns) + 2)
        ws.update([t_str.columns.tolist()] + t_str.values.tolist())
        log.info("Pushed %s (%d rows)", name, len(t_str))


def main() -> int:
    if not os.getenv("OPENAI_API_KEY"):
        log.error("OPENAI_API_KEY missing; copy .env.example to .env")
        return 1
    df = categorize(clean(load()))
    push_to_sheets(aggregate(df))
    log.info("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())