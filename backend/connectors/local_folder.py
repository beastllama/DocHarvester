import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import aiofiles
import pypdf
from docx import Document as DocxDocument
from bs4 import BeautifulSoup
import markdown
import yaml
import json

from backend.config import settings
from .base import BaseConnector, SearchResult


class LocalFolderConnector(BaseConnector):
    """Connector for ingesting documents from local folders"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        folder = config.get("folder_path")
        if not folder:
            raise ValueError("Local folder connector requires a folder_path; none was configured")
        resolved = Path(folder).resolve()
        ingest_root = Path(settings.ingest_root).resolve()
        if not resolved.is_relative_to(ingest_root):
            raise ValueError(
                f"Folder {resolved} is outside the allowed ingest root {ingest_root}"
            )
        self.folder_path = resolved
        self.allowed_extensions = config.get("allowed_extensions", [
            ".txt", ".md", ".pdf", ".docx", ".html", 
            ".json", ".yml", ".yaml", ".py", ".js", ".ts"
        ])
    
    async def test_connection(self) -> bool:
        """Test if the folder exists and is accessible"""
        return self.folder_path.exists() and self.folder_path.is_dir()
    
    async def search(self, query: str, limit: int = 50) -> List[SearchResult]:
        """Search for files in the folder matching the query"""
        results = []
        query_lower = query.lower()
        
        for file_path in self._get_all_files():
            if len(results) >= limit:
                break
                
            # Simple filename and content matching
            if query_lower in file_path.name.lower():
                try:
                    result = await self.fetch_document(str(file_path))
                    if result:
                        results.append(result)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
        
        return results
    
    async def fetch_document(self, doc_id: str) -> SearchResult:
        """Fetch a specific document by file path"""
        file_path = Path(doc_id)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {doc_id}")
        
        # Generate unique doc_id from file path
        doc_hash = hashlib.md5(str(file_path.absolute()).encode()).hexdigest()
        
        # Get file metadata
        stat = file_path.stat()
        last_modified = datetime.fromtimestamp(stat.st_mtime)
        
        # Extract text based on file type
        raw_text = await self._extract_text(file_path)
        
        # Create snippet (first 200 characters)
        snippet = raw_text[:200] + "..." if len(raw_text) > 200 else raw_text
        
        return SearchResult(
            doc_id=doc_hash,
            title=file_path.name,
            snippet=snippet,
            raw_text=raw_text,
            source_type="local_folder",
            source_url=f"file://{file_path.absolute()}",
            source_meta={
                "file_path": str(file_path.absolute()),
                "file_size": stat.st_size,
                "relative_path": str(file_path.relative_to(self.folder_path))
            },
            file_type=file_path.suffix,
            last_modified=last_modified
        )
    
    def _get_all_files(self) -> List[Path]:
        """Get all files in the folder with allowed extensions"""
        files = []
        for ext in self.allowed_extensions:
            files.extend(self.folder_path.rglob(f"*{ext}"))
        return files
    
    async def _extract_text(self, file_path: Path) -> str:
        """Extract text from various file types"""
        extension = file_path.suffix.lower()
        
        try:
            if extension in [".txt", ".py", ".js", ".ts", ".css", ".html", ".xml"]:
                async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                    return await f.read()
            
            elif extension == ".md":
                async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    # Convert markdown to plain text
                    html = markdown.markdown(content)
                    soup = BeautifulSoup(html, 'html.parser')
                    return soup.get_text()
            
            elif extension == ".pdf":
                return self._extract_pdf_text(file_path)
            
            elif extension == ".docx":
                return self._extract_docx_text(file_path)
            
            elif extension in [".yml", ".yaml"]:
                async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    data = yaml.safe_load(content)
                    return yaml.dump(data, default_flow_style=False)
            
            elif extension == ".json":
                async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    data = json.loads(content)
                    return json.dumps(data, indent=2)
            
            else:
                # Try to read as text
                async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                    return await f.read()
                    
        except Exception as e:
            print(f"Error extracting text from {file_path}: {e}")
            return f"Error extracting text: {str(e)}"
    
    def _extract_pdf_text(self, file_path: Path) -> str:
        """Extract text from PDF files"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            text = f"Error extracting PDF text: {str(e)}"
        return text
    
    def _extract_docx_text(self, file_path: Path) -> str:
        """Extract text from DOCX files"""
        try:
            doc = DocxDocument(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except Exception as e:
            return f"Error extracting DOCX text: {str(e)}" 