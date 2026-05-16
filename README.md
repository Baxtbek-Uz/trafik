# Trafik tahlili (Python – yuqori daraja)

## Nima qiladi
Bu loyiha CSV ma’lumotlari asosida:
- **Peak hours** (soat bo‘yicha eng ko‘p qayd/flow)
- **Summary metrikalar** (speed/flow/occupancy uchun: count, mean, median, min, max, std)
- **Anomaly detection** (robust z-score: Median/MAD)
- Natijani **Markdown hisobot**ga chiqaradi.

## Kerakli kutubxonalar
- pandas
- numpy

Masalan:
```bash
pip install pandas numpy
```

## Ishga tushirish
```bash
python traffic_analysis.py --csv <path_to_csv> --time-col time --speed-col speed --flow-col flow --occupancy-col occupancy --anomaly-col flow --out traffic_report.md
```

### Parametrlar
- `--csv` (majburiy): CSV fayl yo‘li
- `--time-col`: vaqt ustuni (default: `time`)
- `--speed-col`: speed ustuni (default: `speed`, bo‘sh qoldirsangiz qiymat ko‘rinmasa e’tiborsiz)
- `--flow-col`: flow ustuni (default: `flow`)
- `--occupancy-col`: occupancy ustuni (default: `occupancy`)
- `--anomaly-col`: anomaliya aniqlash uchun ustun nomi (default: o‘chirilgan)
- `--delimiter`: CSV ajratgich (default: `,`)
- `--out`: hisobot fayli (default: `traffic_report.md`)

## Hisobot
`traffic_report.md` ichida:
- Summary jadvali
- Peak hours (top-10)
- Anomaly detection natijalari (top-30)

