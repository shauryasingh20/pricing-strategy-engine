import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
from sklearn.linear_model import LinearRegression
from fuzzywuzzy import process

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
IMG_DIR = BASE_DIR / "images"

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)
plt.style.use('ggplot')

def load_excel(file_path, year):
    try:
        xl = pd.ExcelFile(file_path)
        sheet_names = xl.sheet_names
        target_sheet = next((s for s in sheet_names if "spending" in s.lower() and "utilization" in s.lower()), sheet_names[0])
        
        df_temp = pd.read_excel(file_path, sheet_name=target_sheet, header=None, nrows=30)
        keywords = {"drug", "brand", "generic", "claims", "spend", "manufacturer", "cost", "total"}
        
        header_row_idx = 0
        max_matches = 0
        
        for i, row in df_temp.iterrows():
            row_str = " ".join(row.astype(str)).lower()
            matches = sum(1 for k in keywords if k in row_str)
            if matches > max_matches:
                max_matches = matches
                header_row_idx = i
                
        df = pd.read_excel(file_path, sheet_name=target_sheet, header=header_row_idx)
        df['year'] = year
        return df
    except Exception as e:
        print(f"Skipping {file_path.name}: {e}")
        return None

def load_data():
    years = range(2016, 2024)
    all_data = []
    for year in years:
        file_path = DATA_RAW / f"partd_{year}.xlsx"
        if file_path.exists():
            df = load_excel(file_path, year)
            if df is not None:
                all_data.append(df)
    
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

def normalize_name(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    patterns = [
        r'\b\d+\s?mg\b', r'\b\d+\s?mcg\b', r'\b\d+\s?ml\b',
        r'\btablet.*', r'\bcapsule.*', r'\binjection.*', r'\bsolution.*',
        r'\bsuspension.*', r'\bcream.*', r'\bointment.*', r'\bpatch.*',
        r'\ber\b', r'\bsr\b', r'\bcr\b', r'\bxr\b', r'[^\w\s]'
    ]
    for p in patterns:
        text = re.sub(p, '', text)
    return " ".join(text.split())

def clean_data(df):
    df.columns = df.columns.astype(str).str.replace('\n', ' ').str.strip()
    
    col_map = {
        "drug_name": ["brand name", "drug name", "product name", "drug"],
        "total_spending": ["total spending", "total drug cost", "gross drug cost"],
        "total_claims": ["total claims", "claim count", "number of claims", "total  claims"],
        "manufacturer_count": ["number of manufacturers", "manufacturer count"]
    }
    
    rename_dict = {}
    for canonical, variants in col_map.items():
        for col in df.columns:
            if str(col).lower().strip() in variants:
                rename_dict[col] = canonical
                break
    df = df.rename(columns=rename_dict)
    
    for col in ["total_spending", "total_claims", "manufacturer_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    if 'total_claims' in df.columns and 'total_spending' in df.columns:
        df['cost_per_claim'] = np.where(df['total_claims'] > 0, df['total_spending'] / df['total_claims'], 0)

    if 'drug_name' in df.columns:
        df['clean_name'] = df['drug_name'].apply(normalize_name)
    
    return df

def link_competitors(df_main):
    orange_book_path = DATA_RAW / "Products.txt"
    if not orange_book_path.exists():
        return df_main.groupby(['clean_name', 'year']).sum(numeric_only=True).reset_index()

    df_ob = pd.read_csv(orange_book_path, sep='~', encoding='latin1', on_bad_lines='skip')
    df_ob.columns = df_ob.columns.str.lower().str.strip()
    
    cols = df_ob.columns
    def find_col(k): return next((c for c in cols if any(x in c for x in k)), None)

    c_prod = find_col(['ingredient', 'name', 'trade_name'])
    c_date = find_col(['approval', 'date'])
    c_type = find_col(['type', 'te_code', 'applicant'])
    
    if not (c_prod and c_date): return df_main

    df_ob['approval_year'] = pd.to_datetime(df_ob[c_date], errors='coerce').dt.year
    df_ob['is_generic'] = df_ob[c_type].astype(str).str.upper().str.contains('ANDA|GENERIC')
    df_ob['clean_name'] = df_ob[c_prod].apply(normalize_name)
    
    first_gen = df_ob[df_ob['is_generic']].groupby('clean_name')['approval_year'].min().reset_index()
    first_gen.rename(columns={'approval_year': 'first_generic_year'}, inplace=True)
    
    merged = pd.merge(df_main, first_gen, on='clean_name', how='left')
    
    unmatched = merged[merged['first_generic_year'].isna()]
    high_spend = unmatched.groupby('clean_name')['total_spending'].sum().nlargest(200).index.tolist()
    gen_names = first_gen['clean_name'].unique()
    
    fuzzy_map = {}
    for name in high_spend:
        match, score = process.extractOne(name, gen_names)
        if score >= 90:
            fuzzy_map[name] = match
            
    if fuzzy_map:
        name_to_year = {k: first_gen.loc[first_gen['clean_name'] == v, 'first_generic_year'].iloc[0] for k, v in fuzzy_map.items()}
        merged['first_generic_year'] = merged['first_generic_year'].fillna(merged['clean_name'].map(name_to_year))
    
    return merged.groupby(['clean_name', 'year']).agg({
        'total_spending': 'sum',
        'total_claims': 'sum',
        'manufacturer_count': 'max',
        'first_generic_year': 'first',
        'cost_per_claim': 'mean'
    }).reset_index()

def generate_charts(df):
    latest_year = df['year'].max()
    
    # 1. Market Structure
    plt.figure(figsize=(12, 6))
    subset = df[df['year'] == latest_year].sort_values('total_spending', ascending=False).head(100)
    sns.scatterplot(data=subset, x='manufacturer_count', y='cost_per_claim', size='total_spending', sizes=(50, 1000), alpha=0.6, legend=False)
    plt.title(f"Market Structure ({latest_year})")
    plt.xlabel("Competitor Count")
    plt.ylabel("Unit Price")
    plt.savefig(IMG_DIR / "market_structure.png")
    plt.close()

    # 2. Lifecycle
    df_life = df.sort_values(['clean_name', 'year']).copy()
    df_life['growth'] = df_life.groupby('clean_name')['total_claims'].pct_change()
    
    def get_stage(g):
        if pd.isna(g): return 'Launch'
        if g > 0.15: return 'Growth'
        if -0.05 <= g <= 0.15: return 'Mature'
        return 'Decline'
    
    df_life['stage'] = df_life['growth'].apply(get_stage)
    top_500 = df[df['year'] == latest_year].sort_values('total_spending', ascending=False).head(500)['clean_name']
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df_life[df_life['clean_name'].isin(top_500)], x='year', hue='stage', multiple='fill', discrete=True, palette='viridis')
    plt.title("Product Lifecycle Composition")
    plt.savefig(IMG_DIR / "product_lifecycle.png")
    plt.close()

    # 3. Forecast
    top_3 = df[df['year'] == latest_year].sort_values('total_spending', ascending=False).head(3)['clean_name'].tolist()
    plt.figure(figsize=(10, 5))
    
    for item in top_3:
        d = df[df['clean_name'] == item].sort_values('year')
        X, y = d[['year']].values, d['total_spending'].values
        model = LinearRegression().fit(X, y)
        fut = np.array([[latest_year+1], [latest_year+2]])
        pred = model.predict(fut)
        
        plt.plot(d['year'], y, marker='o', label=item)
        plt.plot([latest_year, latest_year+1, latest_year+2], [y[-1], pred[0], pred[1]], linestyle='--')
        
    plt.legend()
    plt.title("Demand Forecast")
    plt.savefig(IMG_DIR / "forecast.png")
    plt.close()

    # 4. Elasticity
    df_el = df.sort_values(['clean_name', 'year']).copy()
    df_el['p_chg'] = df_el.groupby('clean_name')['cost_per_claim'].pct_change()
    df_el['v_chg'] = df_el.groupby('clean_name')['total_claims'].pct_change()
    drops = df_el[df_el['p_chg'] < -0.10]
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=drops, x='p_chg', y='v_chg', alpha=0.3)
    plt.title("Price Elasticity")
    plt.savefig(IMG_DIR / "elasticity.png")
    plt.close()

    # 5. Trends
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x='year', y='total_spending', estimator='sum', errorbar=None)
    plt.title("Total GMV Trends")
    plt.savefig(IMG_DIR / "category_trends.png")
    plt.close()

def main():
    df = load_data()
    if df.empty: return

    df = clean_data(df)
    df_final = link_competitors(df)
    
    # Save Data
    df_final.to_parquet(DATA_PROCESSED / "master_marketplace_data.parquet")
    
    # Save Images
    generate_charts(df_final)
    print(f"Processed {len(df_final)} rows. Images saved to {IMG_DIR}")

if __name__ == "__main__":
    main()