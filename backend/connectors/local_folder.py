import os
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple
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

# Obsidian markdown: YAML frontmatter, embeds (![[file]]) and wikilinks ([[Note#Heading|Alias]])
FRONTMATTER = re.compile(r"\A---[ \t]*\n(.*?)\n---[ \t]*(?:\n|\Z)", re.DOTALL)
EMBED = re.compile(r"!\[\[[^\]]*\]\]")
WIKILINK = re.compile(r"\[\[([^\]|#]*)(?:#[^\]|]*)?(?:\|([^\]]*))?\]\]")


def _as_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def parse_markdown_note(content: str) -> Tuple[str, Dict[str, Any]]:
    """Plain text and metadata from a markdown note, including Obsidian syntax.

    Frontmatter is removed from the text; its tags and aliases go into the metadata.
    Wikilinks read as their alias or note name, and their targets are kept as links.
    """
    meta: Dict[str, Any] = {}
    match = FRONTMATTER.match(content)
    if match:
        content = content[match.end():]
        try:
            front = yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError:
            front = {}
        if isinstance(front, dict):
            if front.get("tags"):
                meta["tags"] = _as_list(front["tags"])
            if front.get("aliases"):
                meta["aliases"] = _as_list(front["aliases"])

    links: List[str] = []

    def replace(link: re.Match) -> str:
        target, alias = link.group(1).strip(), link.group(2)
        if target and target not in links:
            links.append(target)
        return (alias or target).strip()

    content = WIKILINK.sub(replace, EMBED.sub("", content))
    if links:
        meta["links"] = links

    html = markdown.markdown(content)
    return BeautifulSoup(html, "html.parser").get_text(), meta


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
        
        # Extract text based on file type. Markdown notes also give tags, aliases and links.
        note_meta: Dict[str, Any] = {}
        if file_path.suffix.lower() == ".md":
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                raw_text, note_meta = parse_markdown_note(await f.read())
            title = file_path.stem
        else:
            raw_text = await self._extract_text(file_path)
            title = file_path.name

        # Create snippet (first 200 characters)
        snippet = raw_text[:200] + "..." if len(raw_text) > 200 else raw_text

        return SearchResult(
            doc_id=doc_hash,
            title=title,
            snippet=snippet,
            raw_text=raw_text,
            source_type="local_folder",
            source_url=f"file://{file_path.absolute()}",
            source_meta={
                "file_path": str(file_path.absolute()),
                "file_size": stat.st_size,
                "relative_path": str(file_path.relative_to(self.folder_path)),
                **note_meta,
            },
            file_type=file_path.suffix,
            last_modified=last_modified
        )
    
    def _get_all_files(self) -> List[Path]:
        """Get all files in the folder with allowed extensions.

        Hidden folders and files are skipped, such as .obsidian/ settings, .trash/ and .git/.
        """
        files = []
        for ext in self.allowed_extensions:
            for path in self.folder_path.rglob(f"*{ext}"):
                if not any(part.startswith(".") for part in path.relative_to(self.folder_path).parts):
                    files.append(path)
        return sorted(files)
    
    async def _extract_text(self, file_path: Path) -> str:
        """Extract text from various file types. Raises if the file cannot be read,
        so a broken file is skipped instead of being saved with an error message as its text."""
        extension = file_path.suffix.lower()

        if extension == ".pdf":
            return self._extract_pdf_text(file_path)
        if extension == ".docx":
            return self._extract_docx_text(file_path)

        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()
        if extension == ".md":
            return parse_markdown_note(content)[0]
        if extension in [".yml", ".yaml"]:
            return yaml.dump(yaml.safe_load(content), default_flow_style=False)
        if extension == ".json":
            return json.dumps(json.loads(content), indent=2)
        return content

    def _extract_pdf_text(self, file_path: Path) -> str:
        """Extract text from PDF files"""
        with open(file_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            return "".join((page.extract_text() or "") + "\n" for page in pdf_reader.pages)

    def _extract_docx_text(self, file_path: Path) -> str:
        """Extract text from DOCX files"""
        doc = DocxDocument(file_path)
        return "\n".join(paragraph.text for paragraph in doc.paragraphs)
