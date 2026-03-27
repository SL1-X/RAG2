from app.config import Config
from app.services.storage.local_storage import LocalStorage
from app.services.storage.minio_storage import MinIOStorage


class StorageFactory:
    _instance = None

    @classmethod
    def create_storage(cls, **kwargs):
        storage_type = getattr(Config, "STORAGE_TYPE", "local").lower()
        if storage_type == "local":
            return LocalStorage()
        elif storage_type == "minio":
            return MinIOStorage()
        else:
            raise ValueError(f"不支持的存储类型:{storage_type}")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls.create_storage()
        return cls._instance
