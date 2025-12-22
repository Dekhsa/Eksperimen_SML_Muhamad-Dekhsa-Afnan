"""
Automated Data Preprocessing for Credit Card Fraud Detection
Author: Muhamad Dekhsa & Afnan
Purpose: Reads raw data and outputs clean preprocessed data
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


class CreditCardPreprocessor:
    """
    Automated preprocessing pipeline for credit card fraud detection dataset
    """
    
    def __init__(self, input_path, output_dir):
        """
        Initialize preprocessor with input and output paths
        
        Args:
            input_path (str): Path to raw CSV file
            output_dir (str): Directory to save preprocessed data
        """
        self.input_path = input_path
        self.output_dir = output_dir
        self.df_raw = None
        self.df_processed = None
        self.scaler = StandardScaler()
        self.le_dict = {}
        
        # Create output directory if not exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def load_data(self):
        """Load raw data from CSV file"""
        print("="*70)
        print("STEP 1: LOADING DATA")
        print("="*70)
        try:
            self.df_raw = pd.read_csv(self.input_path)
            print(f"✓ Data loaded successfully")
            print(f"  Shape: {self.df_raw.shape}")
            print(f"  Columns: {self.df_raw.columns.tolist()}")
        except FileNotFoundError:
            print(f"✗ Error: File not found at {self.input_path}")
            raise
        except Exception as e:
            print(f"✗ Error loading data: {str(e)}")
            raise
    
    def handle_missing_values(self):
        """Handle missing values in the dataset"""
        print("\n" + "="*70)
        print("STEP 2: HANDLING MISSING VALUES")
        print("="*70)
        missing_counts = self.df_raw.isnull().sum()
        if missing_counts.sum() == 0:
            print("✓ No missing values found")
        else:
            print(f"Missing values detected:")
            print(missing_counts[missing_counts > 0])
            # Drop rows with missing values
            self.df_raw = self.df_raw.dropna()
            print(f"✓ Rows with missing values removed. New shape: {self.df_raw.shape}")
    
    def handle_duplicates(self):
        """Remove duplicate rows"""
        print("\n" + "="*70)
        print("STEP 3: HANDLING DUPLICATES")
        print("="*70)
        initial_rows = len(self.df_raw)
        self.df_raw = self.df_raw.drop_duplicates()
        removed_count = initial_rows - len(self.df_raw)
        print(f"✓ Duplicate handling completed")
        print(f"  Rows removed: {removed_count}")
        print(f"  Shape after deduplication: {self.df_raw.shape}")
    
    def handle_outliers(self):
        """Detect and handle outliers using IQR method"""
        print("\n" + "="*70)
        print("STEP 4: HANDLING OUTLIERS (IQR METHOD)")
        print("="*70)
        
        numeric_columns = self.df_raw.select_dtypes(include=[np.number]).columns.tolist()
        # Remove transaction_id from scaling
        numeric_columns = [col for col in numeric_columns if col != 'transaction_id']
        
        print(f"Processing numeric columns: {numeric_columns}")
        
        for col in numeric_columns:
            Q1 = self.df_raw[col].quantile(0.25)
            Q3 = self.df_raw[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers instead of removing
            self.df_raw[col] = self.df_raw[col].clip(lower=lower_bound, upper=upper_bound)
        
        print(f"✓ Outliers capped for all numeric columns")
    
    def feature_binning(self):
        """Create binned features from continuous variables"""
        print("\n" + "="*70)
        print("STEP 5: FEATURE BINNING")
        print("="*70)
        
        # Make a copy for processing
        self.df_processed = self.df_raw.copy()
        
        # Bin amount into categories
        self.df_processed['amount_bin'] = pd.cut(
            self.df_processed['amount'],
            bins=3,
            labels=['Low', 'Medium', 'High'],
            include_lowest=True
        )
        print("Amount bins created:")
        print(self.df_processed['amount_bin'].value_counts().sort_index())
        
        # Bin age into age groups
        self.df_processed['age_group'] = pd.cut(
            self.df_processed['cardholder_age'],
            bins=[0, 25, 35, 50, 65, 100],
            labels=['Youth', 'Young Adult', 'Middle Age', 'Senior', 'Elderly'],
            include_lowest=True
        )
        print("\nAge groups created:")
        print(self.df_processed['age_group'].value_counts().sort_index())
        
        # Bin transaction hour into time periods
        self.df_processed['time_period'] = pd.cut(
            self.df_processed['transaction_hour'],
            bins=[0, 6, 12, 18, 24],
            labels=['Night', 'Morning', 'Afternoon', 'Evening'],
            include_lowest=True,
            right=False
        )
        print("\nTime periods created:")
        print(self.df_processed['time_period'].value_counts().sort_index())
        
        print("\n✓ Feature binning completed")
    
    def encode_categorical_features(self):
        """Encode categorical features using Label Encoding"""
        print("\n" + "="*70)
        print("STEP 6: ENCODING CATEGORICAL FEATURES")
        print("="*70)
        
        # Encode merchant_category
        if 'merchant_category' in self.df_processed.columns:
            le = LabelEncoder()
            self.df_processed['merchant_category_encoded'] = le.fit_transform(
                self.df_processed['merchant_category']
            )
            self.le_dict['merchant_category'] = le
            print("Merchant Category Encoding:")
            for i, label in enumerate(le.classes_):
                print(f"  {label}: {i}")
            self.df_processed = self.df_processed.drop('merchant_category', axis=1)
        
        # Encode binned categorical features
        categorical_bins = ['amount_bin', 'age_group', 'time_period']
        for col in categorical_bins:
            if col in self.df_processed.columns:
                le = LabelEncoder()
                self.df_processed[col + '_encoded'] = le.fit_transform(
                    self.df_processed[col].astype(str)
                )
                self.le_dict[col] = le
                self.df_processed = self.df_processed.drop(col, axis=1)
        
        print(f"✓ All categorical features encoded successfully")
    
    def normalize_features(self):
        """Normalize numeric features using StandardScaler"""
        print("\n" + "="*70)
        print("STEP 7: FEATURE NORMALIZATION/STANDARDIZATION")
        print("="*70)
        
        # Identify columns to scale
        columns_to_scale = [
            'amount', 'transaction_hour', 'device_trust_score',
            'velocity_last_24h', 'cardholder_age', 'merchant_category_encoded'
        ]
        
        # Filter only existing columns
        columns_to_scale = [col for col in columns_to_scale 
                           if col in self.df_processed.columns]
        
        print(f"Scaling columns: {columns_to_scale}")
        
        # Apply StandardScaler
        self.df_processed[columns_to_scale] = self.scaler.fit_transform(
            self.df_processed[columns_to_scale]
        )
        
        print(f"✓ Features normalized using StandardScaler")
        print(f"  Mean ≈ 0, Std ≈ 1")
    
    def remove_unused_columns(self):
        """Remove columns that are not needed for modeling"""
        print("\n" + "="*70)
        print("STEP 8: REMOVING UNUSED COLUMNS")
        print("="*70)
        
        # Columns to drop: ID columns and raw columns that have been encoded
        columns_to_drop = ['transaction_id']
        
        # Also drop if exist and have been encoded
        columns_to_drop.extend([col for col in ['merchant_category', 'amount_bin', 'age_group', 'time_period']
                               if col in self.df_processed.columns])
        
        dropped_cols = [col for col in columns_to_drop if col in self.df_processed.columns]
        self.df_processed = self.df_processed.drop(dropped_cols, axis=1, errors='ignore')
        
        print(f"✓ Columns removed: {dropped_cols}")
        print(f"  Remaining columns: {self.df_processed.columns.tolist()}")
    
    def create_summary(self):
        """Create and display preprocessing summary"""
        print("\n" + "="*70)
        print("PREPROCESSING SUMMARY")
        print("="*70)
        print(f"\nOriginal Dataset Shape: {self.df_raw.shape}")
        print(f"Processed Dataset Shape: {self.df_processed.shape}")
        print(f"\nFeatures in processed dataset:")
        print(f"  Total: {len(self.df_processed.columns)}")
        print(f"  Columns: {self.df_processed.columns.tolist()}")
        print(f"\nData Types:")
        print(self.df_processed.dtypes)
        print(f"\nFirst 5 rows of processed data:")
        print(self.df_processed.head())
        
        # Check target variable distribution if exists
        if 'is_fraud' in self.df_processed.columns:
            print(f"\nTarget Variable Distribution:")
            print(self.df_processed['is_fraud'].value_counts())
            fraud_pct = (self.df_processed['is_fraud'].sum() / len(self.df_processed)) * 100
            print(f"Fraud Rate: {fraud_pct:.2f}%")
    
    def save_data(self, filename='creditcard_clean.csv'):
        """Save preprocessed data to CSV"""
        print("\n" + "="*70)
        print("SAVING CLEANED DATA")
        print("="*70)
        
        output_path = os.path.join(self.output_dir, filename)
        self.df_processed.to_csv(output_path, index=False)
        print(f"✓ Clean data saved successfully")
        print(f"  Path: {output_path}")
        print(f"  Size: {self.df_processed.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    def run_pipeline(self):
        """Execute the complete preprocessing pipeline"""
        print("\n")
        print("╔" + "="*68 + "╗")
        print("║" + " "*15 + "AUTOMATED DATA PREPROCESSING PIPELINE" + " "*16 + "║")
        print("║" + " "*10 + "Credit Card Fraud Detection Dataset" + " "*22 + "║")
        print("╚" + "="*68 + "╝")
        
        try:
            self.load_data()
            self.handle_missing_values()
            self.handle_duplicates()
            self.handle_outliers()
            self.feature_binning()
            self.encode_categorical_features()
            self.normalize_features()
            self.remove_unused_columns()
            self.create_summary()
            self.save_data()
            
            print("\n" + "="*70)
            print("✓ PREPROCESSING COMPLETED SUCCESSFULLY!")
            print("="*70)
            print(f"\nOutput file: {os.path.join(self.output_dir, 'creditcardfraud_preprocessing.csv')}")
            
        except Exception as e:
            print(f"\n✗ ERROR during preprocessing: {str(e)}")
            raise


def main():
    """Main function to run preprocessing"""
    # Get base directory (repository root or current working directory)
    base_dir = os.getenv('GITHUB_WORKSPACE', os.getcwd())
    
    # Define paths - support both local and GitHub Actions environments
    input_file = os.getenv(
        'INPUT_FILE',
        os.path.join(base_dir, 'creditcardfraud_raw.csv')
    )
    
    output_directory = os.getenv(
        'OUTPUT_DIR',
        os.path.join(base_dir, 'preprocessing')
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"✗ Error: Input file not found at {input_file}")
        print(f"  Current working directory: {os.getcwd()}")
        print(f"  Base directory: {base_dir}")
        print(f"  Files in base directory: {os.listdir(base_dir) if os.path.exists(base_dir) else 'N/A'}")
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    print(f"Using input file: {input_file}")
    print(f"Using output directory: {output_directory}")
    
    # Create preprocessor and run pipeline
    preprocessor = CreditCardPreprocessor(input_file, output_directory)
    preprocessor.run_pipeline()


if __name__ == "__main__":
    main()
