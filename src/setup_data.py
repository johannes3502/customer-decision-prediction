"""Simple entry point for preparing the raw datasets."""

from utils.download import (
	ensure_directories,
	download_telco,
	download_bank,
	download_ecom,
)


if __name__ == "__main__":
	ensure_directories()

	telco_path = download_telco()
	print(f"Downloaded: {telco_path}")

	bank_path = download_bank()
	print(f"Downloaded: {bank_path}")

	ecom_path = download_ecom()
	print(f"Downloaded: {ecom_path}")

	print("All datasets downloaded successfully.")
