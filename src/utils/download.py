"""Utilities for downloading the raw project datasets."""

from pathlib import Path
import urllib.request
import zipfile
import io
import shutil


ROOT_DIR = Path(__file__).parent.parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"


def ensure_directories() -> None:
	"""Create the raw and processed data directories.

	Returns:
		None
	"""
	RAW_DIR.mkdir(parents=True, exist_ok=True)
	PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def download_telco() -> Path:
	"""Download the Telco churn dataset if it is not available yet.

	Returns:
		Path to the local CSV file.
	"""
	destination = RAW_DIR / "telco.csv"
	if destination.exists():
		return destination

	url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
	urllib.request.urlretrieve(url, destination)
	return destination


def download_bank() -> Path:
	"""Download the bank marketing dataset from the archive zip.

	Returns:
		Path to the local CSV file.

	Raises:
		FileNotFoundError: If the expected file is missing from the zip archive.
	"""
	destination = RAW_DIR / "bank.csv"
	if destination.exists():
		return destination

	url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
	temp_zip_path = RAW_DIR / "bank_temp.zip"

	urllib.request.urlretrieve(url, temp_zip_path)

	try:
		with zipfile.ZipFile(temp_zip_path, mode="r") as archive:
			expected_member = "bank-additional/bank-additional-full.csv"
			if expected_member not in archive.namelist():
				raise FileNotFoundError(
					f"Expected file '{expected_member}' was not found in the bank archive."
				)

			file_content = archive.read(expected_member)
			destination.write_bytes(file_content)
	finally:
		if temp_zip_path.exists():
			temp_zip_path.unlink()

	return destination


def download_ecom() -> Path:
	"""Download the e-commerce intention dataset if it is missing.

	Returns:
		Path to the local CSV file.
	"""
	destination = RAW_DIR / "ecom.csv"
	if destination.exists():
		return destination

	url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv"
	urllib.request.urlretrieve(url, destination)
	return destination
