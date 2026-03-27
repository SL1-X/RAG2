from app.services.storage.base import StorageInterface
from app.config import Config
import os
from pathlib import Path
from app.utils.logger import get_logger
from minio import Minio
from minio.error import S3Error
from io import BytesIO

logger = get_logger(__name__)


class MinIOStorage(StorageInterface):
    def __init__(self):
        endpoint = getattr(Config, "MINIO_ENDPOINT", "")
        access_key = getattr(Config, "MINIO_ACCESS_KEY", "")
        secret_key = getattr(Config, "MINIO_SECRET_KEY", "")
        bucket_name = getattr(Config, "MINIO_BUCKET_NAME", "")
        secure = getattr(Config, "MINIO_SECURE", "")
        region = getattr(Config, "MINIO_REGION", "")
        self.client = Minio(
            endpoint,
            access_key=access_key,  # 访问密钥 用户名
            secret_key=secret_key,  # 私有密钥 密码
            secure=secure,  # 是否启动HTTPS
            region=region,  # 区域 可选
        )
        self.bucket_name = bucket_name
        # 如果桶不存在，则自动创建桶
        if not self.client.bucket_exists(bucket_name):
            self.client.make_bucket(bucket_name)
            logger.info(f"已创建桶:{bucket_name}")

    def _get_full_path(self, file_path):
        return self.storage_dir / file_path

    def upload_file(self, file_path, file_data):
        try:
            data_stream = BytesIO(file_data)
            self.client.put_object(
                self.bucket_name, file_path, data_stream, length=len(file_data)
            )
            logger.info(f"上传到mino文件{file_path}成功")
        except S3Error as e:
            logger.error(f"上传文件到minio服务器时报错:{e}")
            raise
        except Exception as e:
            logger.error(f"上传文件到minio服务器时报错:{e}")
            raise

    def download_file(self, file_path):
        """
        下载文件

        :param file_path: 文件路径 相对路径

        return: 文件数据bytes
        """
        try:
            # 获取对象句柄
            response = self.client.get_object(self.bucket_name, file_path)
            # 读取对象数据
            data = response.read()
            # 关闭响应
            response.close()
            # 释放连接
            response.release_conn()
            logger.info(f"从mino中读取文件{file_path}成功")
            return data
        except Exception as e:
            logger.error(f"文件下载出错:{e}")
            raise

    def delete_file(self, file_path):
        """
        删除文件

        :param file_path: 文件路径 相对路径
        """
        try:
            self.client.remove_object(self.bucket_name, file_path)
            logger.info(f"从mino中删除文件{file_path}成功")
        except Exception as e:
            logger.error(f"文件下载出错:{e}")
            raise

    def file_exists(self, file_path):
        """
        判断文件是否存在

        :param file_path: 文件路径 相对路径
        """
        pass

    def get_file_url(self, file_path):
        """
        获取文件访问URL地址

        :param file_path: 文件路径 相对路径
        Return 文件URL
        """
        pass
