"""Entry point for downloading and preprocessing all datasets.

This script orchestrates the complete data preparation pipeline by
combining the download utilities with the preprocessing functions from
`utils.preprocess`.

Usage: python src/setup_data.py
"""

from pathlib import Path
import pandas as pd

from utils.download import (
	ensure_directories,
	download_telco,
	download_bank,
	download_ecom,
)

from utils.preprocess import preprocess_data


def main():
	"""Download raw datasets, apply preprocessing, and save clean versions."""
	ensure_directories()
	raw_dir = Path('../data/raw')
	processed_dir = Path('../data/processed')
	processed_dir.mkdir(parents=True, exist_ok=True)

	# Download phase
	print("\n=== DOWNLOAD PHASE ===")
	telco_path = download_telco()
	print(f"Downloaded: {telco_path}")

	bank_path = download_bank()
	print(f"Downloaded: {bank_path}")

	ecom_path = download_ecom()
	print(f"Downloaded: {ecom_path}")

	print("All datasets downloaded successfully.\n")

	# Preprocessing phase
	print("=== PREPROCESSING PHASE ===\n")
	datasets = [
		{
			'name': 'telco',
			'file': raw_dir / 'telco.csv',
			'target_col': 'Churn',
			'target_map': {'Yes': 1, 'No': 0},
			'replace_999_cols': [],
			'read_kwargs': {}
		},
		{
			'name': 'bank',
			'file': raw_dir / 'bank.csv',
			'target_col': 'y',
			'target_map': {'yes': 1, 'no': 0},
			'replace_999_cols': ['pdays'],
			'read_kwargs': {'sep': ';'}
		},
		{
			'name': 'ecom',
			'file': raw_dir / 'ecom.csv',
			'target_col': 'Revenue',
			'target_map': {'True': 1, 'False': 0},
			'replace_999_cols': ['ProductRelated_Duration'],
			'read_kwargs': {}
		}
	]

	for ds in datasets:
		print(f"Processing {ds['name']} ...")
		df = pd.read_csv(ds['file'], **ds.get('read_kwargs', {}))
		print(f"  Shape before cleaning: {df.shape}")
		df_clean = preprocess_data(
			df,
			ds['target_col'],
			ds['target_map'],
			extra_999_cols=ds.get('replace_999_cols', [])
		)
		print(f"  Shape after cleaning: {df_clean.shape}")
		out_path = processed_dir / f"{ds['name']}_clean.csv"
		df_clean.to_csv(out_path, index=False)
		print(f"  Saved to {out_path}\n")

	print("All datasets cleaned and preprocessed successfully.")


if __name__ == "__main__":
	main()
