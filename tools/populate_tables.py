import os
import glob
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
CLEAN_DIR = os.path.join(ROOT, 'data', 'cleaned')
OUT_MD = os.path.join(ROOT, 'tools', 'tables_populated.md')

def summarize():
    files = sorted(glob.glob(os.path.join(CLEAN_DIR, '*.csv')))
    hosp_stats = []
    missing = {}
    outcomes = {}
    total = 0
    for p in files:
        name = os.path.splitext(os.path.basename(p))[0]
        df = pd.read_csv(p)
        n = len(df)
        total += n
        hosp_stats.append((name, n))
        miss = df[['Pulse','Resp','Temp','Sys','Dia']].isna().mean().to_dict()
        missing[name] = {k: float(v) for k,v in miss.items()}
        # outcome counts
        if 'Outcome' in df.columns:
            vc = df['Outcome'].value_counts(dropna=False).to_dict()
            outcomes[name] = {str(k): int(v) for k,v in vc.items()}
        else:
            outcomes[name] = {}

    # write markdown
    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write('### Table 2 — Hospital statistics (per-hospital counts)\n\n')
        f.write('| Hospital | Cleaned records |\n')
        f.write('|---|---:|\n')
        for h,n in hosp_stats:
            f.write(f'| {h} | {n} |\n')
        f.write(f'| **TOTAL** | **{total}** |\n\n')

        f.write('### Table 3 — Missingness summary (per hospital)\n\n')
        f.write('| Hospital | % missing Pulse | % missing Resp | % missing Temp | % missing Sys | % missing Dia |\n')
        f.write('|---|---:|---:|---:|---:|---:|\n')
        for h in missing:
            m = missing[h]
            f.write(f"| {h} | {m.get('Pulse',0)*100:.2f}% | {m.get('Resp',0)*100:.2f}% | {m.get('Temp',0)*100:.2f}% | {m.get('Sys',0)*100:.2f}% | {m.get('Dia',0)*100:.2f}% |\n")
        f.write('\n')

        f.write('### Table 5 — Outcome class counts per hospital\n\n')
        for h in outcomes:
            f.write(f'**{h}**\n\n')
            f.write('| Outcome | Count |\n')
            f.write('|---|---:|\n')
            for k,v in outcomes[h].items():
                f.write(f'| {k} | {v} |\n')
            f.write('\n')

    print('Wrote', OUT_MD)

if __name__ == '__main__':
    summarize()
