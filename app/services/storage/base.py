"""
存储服务抽象接口
"""

from abc import ABC, abstractmethod
from typing import Optional


class StorageInterface(ABC):
    @abstractmethod
    def upload_file(self, file_path, file_data):
        """
        上传文件

        :param file_path: 文件路径 相对路径
        :param file_data: 文件数据(bytes)
        :param content_type: 内容类型

        return: 文件路径
        """

        pass

    @abstractmethod
    def download_file(self, file_path):
        """
        下载文件

        :param file_path: 文件路径 相对路径

        return: 文件数据bytes
        """
        pass

    @abstractmethod
    def delete_file(self, file_path):
        """
        删除文件

        :param file_path: 文件路径 相对路径
        """
        pass

    @abstractmethod
    def file_exists(self, file_path):
        """
        判断文件是否存在

        :param file_path: 文件路径 相对路径
        """
        pass

    @abstractmethod
    def get_file_url(self, file_path):
        """
        获取文件访问URL地址

        :param file_path: 文件路径 相对路径
        Return 文件URL
        """
        pass
