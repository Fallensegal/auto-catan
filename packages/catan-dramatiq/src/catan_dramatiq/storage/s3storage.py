from minio import Minio
from pydantic import SecretStr
from pydantic_settings import BaseSettings

class MinioSettings(BaseSettings):
    minio_user: str = 'admin'
    minio_password: SecretStr = SecretStr('default')


def create_s3_bucket(minio_client: Minio, bucket_name: str) -> str:
    bucket_exist = minio_client.bucket_exists(bucket_name)
    if not bucket_exist:
        minio_client.make_bucket(bucket_name)
        return f"LOG: Success - Created Bucket: {bucket_name}"
    return f"LOG: Info - Bucket Exists: {bucket_name}"

def upload_file_from_fs( minio_client: Minio, bucket_name: str, obj_name: str, file_path: str) -> None:
    minio_client.fput_object(bucket_name, obj_name, file_path)
    

######### Execution ##########

minio_settings = MinioSettings()
minio_client = Minio(
    'chart-minio:9001',
    access_key=minio_settings.minio_user,
    secret_key=minio_settings.minio_password.get_secret_value(),
    secure=False,
)

