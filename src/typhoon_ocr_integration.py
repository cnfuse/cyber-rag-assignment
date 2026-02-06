"""
Typhoon OCR Integration for Thai PDF Processing.

This module provides integration with Typhoon OCR for extracting text
from Thai language PDFs using the specialized Typhoon OCR model.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, List
import json
import requests

from .config import settings

logger = logging.getLogger(__name__)


class TyphoonOCRExtractor:
    """
    Wrapper for Typhoon OCR to extract text from Thai PDFs.
    
    Typhoon OCR is specifically optimized for Thai document extraction
    and provides better results than generic OCR for Thai language PDFs.
    """
    
    def __init__(self):
        """Initialize Typhoon OCR extractor."""
        self.api_key = settings.TYPHOON_API_KEY
        self.base_url = settings.TYPHOON_BASE_URL
        self.model = settings.TYPHOON_OCR_MODEL
        self.enabled = settings.ENABLE_TYPHOON_OCR
        
        if self.enabled:
            if not self.api_key or self.api_key == "your_typhoon_api_key_here":
                logger.warning(
                    "Typhoon OCR is enabled but no valid API key found. "
                    "Set TYPHOON_API_KEY environment variable."
                )
                self.enabled = False
            else:
                logger.info(f"Typhoon OCR enabled with model: {self.model}")
        else:
            logger.info("Typhoon OCR disabled by configuration")
    
    def is_available(self) -> bool:
        """Check if Typhoon OCR is available for use."""
        return self.enabled
    
    def is_thai_pdf(self, pdf_path: Path) -> bool:
        """
        Detect if a PDF is likely in Thai language.
        
        Args:
            pdf_path: Path to the PDF file.
            
        Returns:
            True if the PDF is likely in Thai, False otherwise.
        """
        # Check filename for Thai indicators
        filename_lower = pdf_path.stem.lower()
        thai_keywords = [
            'thai', 'thailand', 'ไทย', 'ประเทศไทย',
            'กระทรวง', 'มาตรฐาน', 'ภาษาไทย'
        ]
        
        for keyword in thai_keywords:
            if keyword in filename_lower:
                logger.info(f"Detected Thai PDF by filename: {pdf_path.name}")
                return True
        
        return False
    
    def extract_text(
        self,
        pdf_path: Path,
        page_idx: int = 0,
        task_type: str = "default",
        max_tokens: int = 16384,
        temperature: float = 0.1,
        top_p: float = 0.6,
        repetition_penalty: float = 1.2
    ) -> Optional[str]:
        """
        Extract text from a specific PDF page using Typhoon OCR API.
        
        Args:
            pdf_path: Path to the PDF file.
            page_idx: Page index (0-indexed).
            task_type: OCR task type ("default", "v1.5" for clean Markdown, etc.).
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            repetition_penalty: Repetition penalty.
            
        Returns:
            Extracted text, or None if extraction fails.
        """
        if not self.enabled:
            return None
        
        try:
            # Convert 0-indexed to 1-indexed for API
            page_num = page_idx + 1
            
            logger.info(
                f"Extracting text from {pdf_path.name} page {page_num} "
                f"using Typhoon OCR API"
            )
            
            url = f"{self.base_url}/ocr"
            
            with open(pdf_path, 'rb') as file:
                files = {'file': file}
                data = {
                    'model': self.model,
                    'task_type': task_type,
                    'max_tokens': str(max_tokens),
                    'temperature': str(temperature),
                    'top_p': str(top_p),
                    'repetition_penalty': str(repetition_penalty),
                    'pages': json.dumps([page_num])  # Single page
                }
                
                headers = {
                    'Authorization': f'Bearer {self.api_key}'
                }
                
                response = requests.post(url, files=files, data=data, headers=headers, timeout=60)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Extract text from successful results
                    extracted_texts = []
                    for page_result in result.get('results', []):
                        if page_result.get('success') and page_result.get('message'):
                            content = page_result['message']['choices'][0]['message']['content']
                            try:
                                # Try to parse as JSON if it's structured output
                                parsed_content = json.loads(content)
                                text = parsed_content.get('natural_text', content)
                            except json.JSONDecodeError:
                                text = content
                            extracted_texts.append(text)
                        elif not page_result.get('success'):
                            error_msg = page_result.get('error', 'Unknown error')
                            logger.warning(
                                f"Error processing page {page_num}: {error_msg}"
                            )
                    
                    if extracted_texts:
                        full_text = '\n'.join(extracted_texts)
                        logger.info(
                            f"Successfully extracted {len(full_text)} characters "
                            f"from page {page_num}"
                        )
                        return full_text
                    else:
                        logger.warning(
                            f"No text extracted from page {page_num}"
                        )
                        return None
                else:
                    logger.error(
                        f"Typhoon OCR API error: {response.status_code} - {response.text}"
                    )
                    return None
                    
        except Exception as e:
            logger.error(
                f"Typhoon OCR extraction failed for {pdf_path.name} "
                f"page {page_idx + 1}: {e}",
                exc_info=True
            )
            return None
    
    def extract_page(
        self,
        pdf_path: Path,
        page_num: int,
        task_type: str = "default"
    ) -> Optional[str]:
        """
        Extract text from a specific PDF page using Typhoon OCR.
        
        Args:
            pdf_path: Path to the PDF file.
            page_num: Page number (1-indexed).
            task_type: OCR task type ("default", "v1.5" for clean Markdown, etc.).
            
        Returns:
            Extracted text, or None if extraction fails.
        """
        # Convert 1-indexed to 0-indexed for extract_text
        return self.extract_text(pdf_path, page_num - 1, task_type)
    
    def extract_all_pages(
        self,
        pdf_path: Path,
        max_pages: Optional[int] = None,
        task_type: str = "default"
    ) -> Dict[int, str]:
        """
        Extract text from all pages of a PDF using Typhoon OCR.
        
        Args:
            pdf_path: Path to the PDF file.
            max_pages: Maximum number of pages to process (None for all).
            task_type: OCR task type.
            
        Returns:
            Dictionary mapping page numbers (1-indexed) to extracted text.
        """
        if not self.enabled:
            return {}
        
        results = {}
        
        # Try to get page count
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(pdf_path))
            total_pages = len(reader.pages)
            
            if max_pages:
                total_pages = min(total_pages, max_pages)
            
            logger.info(
                f"Processing {total_pages} pages from {pdf_path.name} "
                f"with Typhoon OCR"
            )
            
            for page_num in range(1, total_pages + 1):
                extracted_text = self.extract_page(pdf_path, page_num, task_type)
                if extracted_text:
                    results[page_num] = extracted_text
                    
        except Exception as e:
            logger.error(
                f"Failed to extract all pages from {pdf_path.name}: {e}",
                exc_info=True
            )
        
        return results
    


# Global instance
_typhoon_ocr_extractor: Optional[TyphoonOCRExtractor] = None


def get_typhoon_ocr_extractor() -> TyphoonOCRExtractor:
    """
    Get the global Typhoon OCR extractor instance.
    
    Returns:
        TyphoonOCRExtractor instance.
    """
    global _typhoon_ocr_extractor
    if _typhoon_ocr_extractor is None:
        _typhoon_ocr_extractor = TyphoonOCRExtractor()
    return _typhoon_ocr_extractor
