"""
pull_fg_undetected.py - undetected-chromedriver to bypass Cloudflare's headless detection.
Runs visibly (not headless) so Cloudflare's challenge is solvable.
"""
from __future__ import annotations
import sys, time, re, json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / 'data' / 'outputs'

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]

PITCHER_COLS = {
    'xMLBAMID':'mlb_id','Name':'player_name_fg','Team':'team','Season':'season',
    'IP':'ip','G':'g','GS':'gs','ERA':'era','FIP':'fip','xFIP':'xfip','SIERA':'siera',
    'xERA':'xera','WHIP':'whip','K%':'k_pct','BB%':'bb_pct','K-BB%':'k_minus_bb_pct',
    'SwStr%':'swstr_pct','CSW%':'csw_pct','C+SwStr%':'c_plus_swstr_pct','HR/FB':'hr_fb',
    'GB%':'gb_pct','LOB%':'lob_pct','Barrel%':'barrel_pct','HardHit%':'hard_hit_pct',
    'EV':'avg_ev','sp_stuff':'stuff_plus','sp_location':'location_plus',
    'sp_pitching':'pitching_plus','pb_stuff':'pb_stuff','pb_command':'pb_command',
    'pb_xRV100':'pb_xrv100',
}

def clean_name(raw):
    if not isinstance(raw, str): return str(raw)
    return re.sub(r'<[^>]+>', '', raw).strip()


def fetch_year(driver, year: int):
    """Fetch year via JSON API call inside browser session (Cloudflare cookies set)."""
    api_url = (f'https://www.fangraphs.com/api/leaders/major-league/data?'
                f'pos=all&stats=pit&lg=all&qual=10&season={year}&season1={year}'
                f'&month=0&team=0&pageitems=500&pagenum=1&ind=0&type=8')

    # First, navigate to leaderboard page to get Cloudflare cookies
    leaderboard_url = f'https://www.fangraphs.com/leaders/major-league?stats=pit&season={year}&type=8'
    print(f'  [{year}] navigating to leaderboard page (warming session)...', flush=True)
    driver.get(leaderboard_url)
    time.sleep(8)  # let CF challenge complete + page render

    # Trigger fetch in browser context
    print(f'  [{year}] fetching API via browser fetch()...', flush=True)
    js = f'''
    return await new Promise((resolve) => {{
        fetch("{api_url}", {{credentials: "include"}})
            .then(r => r.text().then(t => resolve({{status: r.status, body: t}})))
            .catch(e => resolve({{status: 0, body: String(e)}}));
    }});
    '''
    response = driver.execute_script(js)
    status = response.get('status')
    if status != 200:
        return None, f'HTTP {status}; body[:200]={response.get("body","")[:200]}'

    try:
        data = json.loads(response['body'])
    except Exception as e:
        return None, f'JSON parse failed: {e}'

    rows = data.get('data', [])
    if not rows: return None, 'no data rows'

    records = []
    for row in rows:
        rec = {}
        for src, dst in PITCHER_COLS.items():
            v = row.get(src)
            if isinstance(v, str) and '<' in v:
                v = clean_name(v)
            rec[dst] = v
        records.append(rec)
    df = pd.DataFrame(records)
    if 'mlb_id' in df.columns:
        df['mlb_id'] = pd.to_numeric(df['mlb_id'], errors='coerce').astype('Int64')
    pct_cols = [c for c in df.columns if c.endswith('_pct')]
    for c in pct_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        if df[c].dropna().between(0, 1).all():
            df[c] = (df[c] * 100).round(2)
    return df, None


def main():
    import undetected_chromedriver as uc

    print('Launching undetected Chrome (visible, not headless)...', flush=True)
    options = uc.ChromeOptions()
    # Run NOT headless — Cloudflare detects headless mode
    # options.add_argument('--headless=new')  # SKIP
    options.add_argument('--disable-blink-features=AutomationControlled')
    driver = uc.Chrome(options=options, version_main=147)  # match installed Chrome 147

    successes = 0
    try:
        for yr in YEARS:
            out_path = OUTPUTS / f'fangraphs_pitchers_{yr}.csv'
            if out_path.exists():
                ex = pd.read_csv(out_path)
                if len(ex) > 100 and 'stuff_plus' in ex.columns and ex['stuff_plus'].notna().sum() > 50:
                    print(f'[{yr}] cached -> skipping')
                    continue

            try:
                df, err = fetch_year(driver, yr)
                if df is not None and len(df) > 50:
                    stuff_n = df['stuff_plus'].notna().sum() if 'stuff_plus' in df.columns else 0
                    df.to_csv(out_path, index=False)
                    print(f'[{yr}] SUCCESS: {len(df)} rows, stuff_plus={stuff_n} populated')
                    successes += 1
                else:
                    print(f'[{yr}] FAILED: {err}')
            except Exception as e:
                print(f'[{yr}] EXCEPTION: {e}')
            time.sleep(3)
    finally:
        try: driver.quit()
        except Exception: pass

    print(f'\n=== {successes}/{len(YEARS)} years pulled ===', flush=True)


if __name__ == '__main__':
    main()
