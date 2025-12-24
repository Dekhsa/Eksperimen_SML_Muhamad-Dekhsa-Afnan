# Eksperimen SML - Credit Card Fraud Detection

Automated data preprocessing pipeline untuk dataset credit card fraud detection.

## Struktur Project

```
.
├── creditcardfraud_raw.csv              # Raw input data
├── requrements.txt                      # Python dependencies
├── README.md                            # Project documentation
├── preprocessing/
│   ├── automate_Muhamad-Dekhsa-Afnan.py # Automated preprocessing script
│   ├── Eksperimen_Muhamad-Dekhsa-Afnan.ipynb  # Jupyter notebook untuk analisis & eksplorasi
│   └── creditcardfraud_preprocess.csv   # Preprocessed data output
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

Hasil preprocessing disimpan di `creditcardfraud_preprocess.csv` dengan:
- **Features yang sudah dinormalisasi** - Menggunakan StandardScaler
- **Categorical features yang sudah di-encode** - Label encoding untuk merchant_category dan binned features
- **Binned features**:
  - `amount_bin_encoded` - Amount binning (Low, Medium, High)
  - `age_group_encoded` - Age grouping (Youth, Young Adult, Middle Age, Senior, Elderly)
  - `time_period_encoded` - Time period (Night, Morning, Afternoon, Evening)
- **Original dataset shape vs processed shape** - Informasi transformasi data
- **Target variable distribution** - Statistik fraud dan non-fraud cases

## Jalankan di Jupyter Notebook

1. Buka Jupyter Notebook
```bash
cd preprocessing
jupyter notebook Eksperimen_Muhamad-Dekhsa-Afnan.ipynb
```

2. Jalankan semua cells dengan Ctrl+A kemudian Shift+Enter

3. File preprocessed akan disimpan sebagai `creditcardfraud_preprocess.csv`

## GitHub Actions

Pipeline otomatis berjalan ketika ada push ke branch `main`:

1. Setup Python 3.12.7
2. Install dependencies dari `requrements.txt`
3. Jalankan preprocessing script
4. Upload hasil preprocessing (`creditcardfraud_preprocess.csv`) ke artifacts

Setiap workflow run menghasilkan artifact yang dapat didownload dari GitHub.

### Environment Variables

Dapat dikonfigurasi melalui GitHub Actions:
- `INPUT_FILE` - Path ke raw CSV (default: `creditcardfraud_raw.csv`)
- `OUTPUT_DIR` - Directory untuk output (default: `preprocessing/`)

## Authors

- Muhamad Dekhsa
- Afnan

## Dataset Source

Dataset diunduh dari: [Kaggle - Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/miadul/credit-card-fraud-detection-dataset)

**Dataset Overview:**
- Total Records: 10,000
- Features: 9 Predictors + 1 Target
- Target Variable: `is_fraud` (Binary: 0 = Normal, 1 = Fraud)
- Class Distribution: ~4-5% Fraud (Highly Imbalanced)

**Features:**
- `transaction_id` - Unique transaction identifier
- `amount` - Transaction amount
- `transaction_hour` - Hour of transaction (0-23)
- `merchant_category` - Type of merchant
- `foreign_transaction` - International (1) or domestic (0)
- `location_mismatch` - Location mismatch indicator
- `device_trust_score` - Device trust score (0-100)
- `velocity_last_24h` - Transaction count in last 24 hours
- `cardholder_age` - Age of cardholder
- `is_fraud` - Target variable (0 = Normal, 1 = Fraud)

## License

MIT
