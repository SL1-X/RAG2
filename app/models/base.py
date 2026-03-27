from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.inspection import inspect

# 创建统一的Base类，所有的ORM模型都应该继承此Base类
Base = declarative_base()


# 定义所有的模型的基类
class BaseModel(Base):
    # 把此类标准为抽象类，这样就不会创建表了
    __abstract__ = True

    # 把模型对象转成python字典的方法
    def to_dict(self, exclude=[], **kwargs):
        # 要排除的字段列表 ["password_hash"]
        result = {}
        # 获取当模型类的所有的列定义
        mapper = inspect(self.__class__)
        # 遍历所有的列
        for column in mapper.columns:
            # 获取列名
            col_name = column.name
            if col_name in exclude:
                continue
            # 获取此字段名的值
            value = getattr(self, col_name, None)
            # 如果value有是日期类型的话，调用isoformat转换为字符串
            if hasattr(value, "isoformat"):
                result[col_name] = value.isoformat() if value else None
            else:
                result[col_name] = value
        return result

    def __repr__(self):
        # 如果子类指定义__repr_fields__值，优先显示这些字段
        if hasattr(self, "__repr_fields__"):
            fields = getattr(self, "__repr_fields__")
            attrs = ", ".join(f"{field}={getattr(self,field,None)}" for field in fields)
        else:
            attrs = f"id={getattr(self,'id',None)}"
        return f"<{self.__class__.__name__}({attrs})>"
