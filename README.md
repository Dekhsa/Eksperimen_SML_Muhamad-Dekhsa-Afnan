# Eksperimen SML - Credit Card Fraud Detection

Automated data preprocessing pipeline untuk dataset credit card fraud detection.

## Struktur Project

```
.
├── creditcardfraud_raw.csv              # Raw input data
├── requrements.txt                      # Python dependencies
├── preprocessing/
│   ├── automate_Muhamad-Dekhsa-Afnan.py # Main preprocessing script
│   └── Eksperimen_MSML.ipynb            # Jupyter notebook untuk analisis
└── .github/workflows/
    └── preprocessing.yml                # GitHub Actions workflow
```

## Requirements

- Python 3.12.7+
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 0.24.0

## Setup & Installation

### Local Development

1. Clone repository
```bash
git clone https://github.com/Dekhsa/Eksperimen_SML_Muhamad-Dekhsa-Afnan.git
cd Eksperimen_SML_Muhamad-Dekhsa-Afnan
```

2. Install dependencies
```bash
pip install -r requrements.txt
```

3. Jalankan preprocessing pipeline
```bash
cd preprocessing
python automate_Muhamad-Dekhsa-Afnan.py
```

## Preprocessing Pipeline

Pipeline melakukan preprocessing otomatis dengan langkah-langkah berikut:

1. **Loading Data** - Membaca raw CSV file
2. **Handling Missing Values** - Menghapus rows dengan nilai kosong
3. **Handling Duplicates** - Menghilangkan duplicate rows
4. **Handling Outliers** - Capping outliers menggunakan IQR method
5. **Feature Binning** - Membuat binned features dari continuous variables:
   - Amount binning (Low, Medium, High)
   - Age grouping (Youth, Young Adult, Middle Age, Senior, Elderly)
   - Time period binning (Night, Morning, Afternoon, Evening)
6. **Encoding Categorical Features** - Label encoding untuk categorical variables
7. **Feature Normalization** - Standardisasi numeric features
8. **Removing Unused Columns** - Menghapus transaction_id dan raw columns
9. **Data Summary** - Menampilkan overview preprocessed data

## Output

Hasil preprocessing disimpan di `preprocessing/creditcard_clean.csv` dengan:
- Features yang sudah dinormalisasi
- Categorical features yang sudah di-encode
- Original dataset shape vs processed shape
- Target variable distribution (jika ada)

## GitHub Actions

Pipeline otomatis berjalan ketika ada push ke branch `main`:

1. Setup Python 3.12.7
2. Install dependencies
3. Run preprocessing pipeline
4. Upload hasil ke artifacts

### Environment Variables

Dapat dikonfigurasi melalui GitHub Actions:
- `INPUT_FILE` - Path ke raw CSV (default: `creditcardfraud_raw.csv`)
- `OUTPUT_DIR` - Directory untuk output (default: `preprocessing/`)

## Authors

- Muhamad Dekhsa
- Afnan

## License

MIT
