import os
from pathlib import Path
import json
import shutil
import sys

# Add parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, RESULTS_DIR, AWS_S3_BUCKET, AWS_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY

try:
    import boto3
    from botocore.exceptions import BotoCoreError, NoCredentialsError, ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

class S3Manager:
    """
    AWS S3 Cloud Storage Manager with automatic fallback to Local Mode
    if AWS credentials or boto3 library are unavailable.
    """
    def __init__(self, bucket_name=AWS_S3_BUCKET, region=AWS_REGION, access_key=AWS_ACCESS_KEY_ID, secret_key=AWS_SECRET_ACCESS_KEY):
        self.bucket_name = bucket_name
        self.region = region
        self.access_key = access_key
        self.secret_key = secret_key
        self.s3_client = None
        self.is_connected = False
        self.local_mode = True
        self.status_message = "Local Mode Active (No AWS Credentials provided)"

        self._initialize_connection()

    def _initialize_connection(self):
        """Attempts connection to AWS S3, falls back to local storage if unavailable."""
        if not BOTO3_AVAILABLE:
            self.local_mode = True
            self.status_message = "Local Mode: boto3 library not installed."
            return

        if not self.access_key or not self.secret_key:
            self.local_mode = True
            self.status_message = "Local Mode Active: AWS Access Keys not configured."
            return

        try:
            self.s3_client = boto3.client(
                's3',
                region_name=self.region,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key
            )
            # Test connection by checking bucket
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            self.is_connected = True
            self.local_mode = False
            self.status_message = f"Connected to AWS S3 Bucket: {self.bucket_name}"
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == '404':
                # Bucket does not exist, attempt to create it
                try:
                    if self.region == 'us-east-1':
                        self.s3_client.create_bucket(Bucket=self.bucket_name)
                    else:
                        self.s3_client.create_bucket(
                            Bucket=self.bucket_name,
                            CreateBucketConfiguration={'LocationConstraint': self.region}
                        )
                    self.is_connected = True
                    self.local_mode = False
                    self.status_message = f"Created & Connected to AWS S3 Bucket: {self.bucket_name}"
                except Exception as create_err:
                    self.local_mode = True
                    self.status_message = f"AWS S3 Creation Failed: {str(create_err)}. Operating in Local Mode."
            else:
                self.local_mode = True
                self.status_message = f"AWS S3 Access Error ({error_code}). Operating in Local Mode."
        except Exception as e:
            self.local_mode = True
            self.status_message = f"AWS Connection Failed ({str(e)}). Operating in Local Mode."

    def upload_dataset(self, local_path: Path, s3_key: str = None) -> str:
        """
        Uploads dataset CSV to S3 bucket or copies to data/ folder in local mode.
        """
        local_path = Path(local_path)
        if not s3_key:
            s3_key = f"uploads/{local_path.name}"

        if not self.local_mode and self.s3_client:
            try:
                self.s3_client.upload_file(str(local_path), self.bucket_name, s3_key)
                return f"s3://{self.bucket_name}/{s3_key}"
            except Exception as e:
                print(f"S3 Upload failed: {e}. Falling back to local copy.")

        # Local Mode Upload
        target_path = DATA_DIR / local_path.name
        if local_path.resolve() != target_path.resolve():
            shutil.copy(local_path, target_path)
        return str(target_path)

    def upload_result(self, content_str: str, filename: str, folder: str = "processed-results") -> str:
        """
        Saves result text/JSON/CSV into S3 bucket or local results/ folder.
        """
        s3_key = f"{folder}/{filename}"
        
        if not self.local_mode and self.s3_client:
            try:
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=s3_key,
                    Body=content_str
                )
                return f"s3://{self.bucket_name}/{s3_key}"
            except Exception as e:
                print(f"S3 Result Save failed: {e}. Writing to local results.")

        # Local Mode Save
        target_path = RESULTS_DIR / filename
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(content_str)
        return str(target_path)

    def list_files(self) -> dict:
        """
        Returns list of stored files in Cloud S3 or Local directories.
        """
        if not self.local_mode and self.s3_client:
            try:
                res = self.s3_client.list_objects_v2(Bucket=self.bucket_name)
                contents = res.get('Contents', [])
                return {
                    "mode": "AWS S3 Cloud",
                    "files": [obj['Key'] for obj in contents]
                }
            except Exception as e:
                pass

        # Local Mode listing
        local_data_files = [f"uploads/{f.name}" for f in DATA_DIR.glob("*") if f.is_file()]
        local_result_files = [f"processed-results/{f.name}" for f in RESULTS_DIR.glob("*") if f.is_file()]
        return {
            "mode": "Local Storage",
            "files": local_data_files + local_result_files
        }
