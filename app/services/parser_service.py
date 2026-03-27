from app.utils.document_loader import DocumentLoader


class ParserService:
    def parse(self, file_data, file_type):
        return DocumentLoader.load(file_data, file_type)


parser_service = ParserService()
