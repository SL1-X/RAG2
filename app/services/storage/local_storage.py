from app.services.storage.base import StorageInterface
from app.config import Config
import os
from pathlib import Path
from app.utils.logger import get_logger

logger = get_logger(__name__)


class LocalStorage(StorageInterface):
    def __init__(self):
        storage_dir = Config.STORAGE_DIR
        # 判断storage_dir是否是绝对路径
        if os.path.isabs(storage_dir):
            self.storage_dir = Path(storage_dir)
        else:
            base_dir = Path(__file__).parent.parent.parent.parent
            self.storage_dir = base_dir / storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"本地存储初始化成功：目录：{self.storage_dir}")

    def _get_full_path(self, file_path):
        return self.storage_dir / file_path

    def upload_file(self, file_path, file_data):
        full_path = self._get_full_path(file_path)
        # 保证文件路径的父目录是存在
        full_path.parent.mkdir(parents=True, exist_ok=True)
        # 把文件内容写入文件路径
        with open(full_path, "wb") as f:
            f.write(file_data)
        logger.info(f"文件已经上传:{file_path}")
        return file_path

    def download_file(self, file_path):
        """
        下载文件

        :param file_path: 文件路径 相对路径

        return: 文件数据bytes
        """
        try:
            full_path = self._get_full_path(file_path)
            with open(full_path, "rb") as f:
                data = f.read()
            logger.info(f"文件已经下载成功:{full_path}")
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
            full_path = self._get_full_path(file_path)
            if full_path.exists():
                full_path.unlink()
            logger.info(f"文件已经下删除:{full_path}")
            try:
                # 尝试删除父目录，因为如果当前的文件删除后父目录为空了，则可以删除，否则 其实删除会失败
                full_path.parent.rmdir()
            except OSError:
                pass
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
