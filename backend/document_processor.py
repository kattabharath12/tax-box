import os
import re
import logging
from typing import Dict, Any, List
from datetime import datetime

# Try to import OCR libraries (graceful fallback if not available)
try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    from pdf2image import convert_from_path
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        # Initialize EasyOCR reader if available
        self.easyocr_reader = None
        if EASYOCR_AVAILABLE:
            try:
                self.easyocr_reader = easyocr.Reader(['en'])
                logger.info("✅ EasyOCR initialized successfully")
            except Exception as e:
                logger.warning(f"⚠️ EasyOCR initialization failed: {e}")
        
        # Tax document patterns for extraction
        self.tax_patterns = {
            'w2': {
                'wages': [
                    r'wages.*?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
                    r'box\s*1.*?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
                    r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?).*wages'
                ],
                'federal_tax': [
                    r'federal.*?tax.*?withheld.*?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
                    r'box\s*2.*?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
                    r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?).*federal.*tax'
                ],
                'employer': [
                    r'employer.*?name.*?([A-Za-z\s&,.-]+)',
                    r'company.*?([A-Za-z\s&,.-]+)',
                ]
            },
            '1099': {
                'income': [
                    r'nonemployee.*?compensation.*?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
                    r'box\s*1.*?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
                    r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?).*compensation'
                ],
                'payer': [
                    r'payer.*?([A-Za-z\s&,.-]+)',
                    r'company.*?([A-Za-z\s&,.-]+)',
                ]
            }
        }

    def process_document(self, file_path: str, file_type: str) -> Dict[str, Any]:
        """Main method to process uploaded documents with basic OCR"""
        try:
            logger.info(f"Processing document: {file_path}")
            
            # Step 1: Extract text from document
            text = self._extract_text_from_document(file_path, file_type)
            
            if not text:
                return {"error": "Could not extract text from document"}
            
            # Step 2: Classify document type
            doc_type = self._classify_document_type(text, file_path)
            
            # Step 3: Extract structured data
            extracted_data = self._extract_structured_data(text, doc_type)
            
            return {
                "success": True,
                "document_type": doc_type,
                "extracted_data": extracted_data,
                "confidence": self._calculate_confidence(extracted_data, doc_type),
                "raw_text": text[:500],  # First 500 chars for debugging
                "processing_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return {"error": f"Processing failed: {str(e)}"}

    def _extract_text_from_document(self, file_path: str, file_type: str) -> str:
        """Extract text using multiple methods with fallbacks"""
        text = ""
        
        try:
            # Method 1: Image files with OCR
            if file_type.lower() in ['jpg', 'jpeg', 'png', 'tiff', 'bmp']:
                text = self._extract_from_image(file_path)
            
            # Method 2: PDF files
            elif file_type.lower() == 'pdf':
                text = self._extract_from_pdf(file_path)
            
            # Method 3: Word documents
            elif file_type.lower() in ['doc', 'docx']:
                text = self._extract_from_docx(file_path)
            
            # Method 4: Plain text fallback
            else:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                except:
                    with open(file_path, 'r', encoding='latin-1') as f:
                        text = f.read()
                        
        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            
        return text.strip()

    def _extract_from_image(self, file_path: str) -> str:
        """Extract text from image using OCR"""
        text = ""
        
        # Try EasyOCR first (usually better results)
        if self.easyocr_reader:
            try:
                results = self.easyocr_reader.readtext(file_path)
                text = ' '.join([result[1] for result in results])
                logger.info("✅ EasyOCR extraction successful")
            except Exception as e:
                logger.warning(f"EasyOCR failed: {e}")
        
        # Fallback to Tesseract
        if not text and OCR_AVAILABLE:
            try:
                image = Image.open(file_path)
                text = pytesseract.image_to_string(image)
                logger.info("✅ Tesseract extraction successful")
            except Exception as e:
                logger.warning(f"Tesseract failed: {e}")
        
        return text

    def _extract_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF"""
        text = ""
        
        if not PDF_AVAILABLE:
            return text
        
        # Try text extraction first
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text()
        except Exception as e:
            logger.warning(f"PDF text extraction failed: {e}")
        
        # If no text found, convert to images and OCR
        if not text.strip():
            try:
                images = convert_from_path(file_path)
                for image in images:
                    text += self._extract_from_image_object(image)
            except Exception as e:
                logger.warning(f"PDF to image conversion failed: {e}")
        
        return text

    def _extract_from_image_object(self, image) -> str:
        """Extract text from PIL Image object"""
        text = ""
        
        # Try EasyOCR
        if self.easyocr_reader:
            try:
                import numpy as np
                img_array = np.array(image)
                results = self.easyocr_reader.readtext(img_array)
                text = ' '.join([result[1] for result in results])
            except Exception as e:
                logger.warning(f"EasyOCR on image object failed: {e}")
        
        # Fallback to Tesseract
        if not text and OCR_AVAILABLE:
            try:
                text = pytesseract.image_to_string(image)
            except Exception as e:
                logger.warning(f"Tesseract on image object failed: {e}")
        
        return text

    def _extract_from_docx(self, file_path: str) -> str:
        """Extract text from Word documents"""
        text = ""
        
        if not DOCX_AVAILABLE:
            return text
        
        try:
            doc = DocxDocument(file_path)
            text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            logger.warning(f"DOCX extraction failed: {e}")
        
        return text

    def _classify_document_type(self, text: str, filename: str) -> str:
        """Classify document type based on content and filename"""
        text_lower = text.lower()
        filename_lower = filename.lower()
        
        # Check filename first
        if any(keyword in filename_lower for keyword in ['w-2', 'w2']):
            return 'w2'
        elif any(keyword in filename_lower for keyword in ['1099']):
            return '1099'
        
        # Check content
        if any(keyword in text_lower for keyword in ['wage and tax statement', 'w-2', 'employer identification']):
            return 'w2'
        elif any(keyword in text_lower for keyword in ['1099-misc', '1099-nec', 'nonemployee compensation']):
            return '1099'
        elif any(keyword in text_lower for keyword in ['1098', 'mortgage interest']):
            return '1098'
        
        return 'unknown'

    def _extract_structured_data(self, text: str, doc_type: str) -> Dict[str, Any]:
        """Extract structured data using improved pattern matching"""
        extracted = {}
        
        if doc_type == 'w2':
            # More flexible patterns for W-2
            patterns = {
                'wages': [
                    r'1\.\s*wages.*?[\$]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
                    r'wages.*?compensation.*?[\$]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
                    r'[\$]\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?).*wages',
                    r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*.*wages'
                ],
                'federal_tax': [
                    r'2\.\s*federal.*?tax.*?[\$]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
                    r'federal.*?tax.*?withheld.*?[\$]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
                    r'[\$]\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?).*federal.*tax',
                    r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*.*federal.*tax'
                ],
                'employer': [
                    r"employer's\s*name.*?:\s*([A-Za-z\s&,.-]+)",
                    r"employer.*?([A-Z][A-Za-z\s&,.-]{3,})",
                ]
            }
        elif doc_type == '1099':
            patterns = {
                'income': [
                    r'1\.\s*nonemployee.*?compensation.*?[\$]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
                    r'nonemployee.*?compensation.*?[\$]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
                    r'[\$]\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?).*compensation',
                    r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*.*compensation'
                ]
            }
        else:
            return extracted
        
        # Apply patterns
        for field, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
                if matches:
                    value = matches[0].strip()
                    
                    # Clean and convert numeric values
                    if field in ['wages', 'income', 'federal_tax']:
                        clean_value = re.sub(r'[^\d.]', '', value)
                        try:
                            extracted[field] = float(clean_value) if clean_value else 0
                            print(f"✅ Extracted {field}: {extracted[field]}")
                            break  # Stop at first successful match
                        except ValueError:
                            continue
                    else:
                        # Clean text values
                        clean_text = re.sub(r'[^\w\s&,.-]', '', value).strip()
                        if len(clean_text) > 2:  # Only accept meaningful text
                            extracted[field] = clean_text
                            print(f"✅ Extracted {field}: {extracted[field]}")
                            break
        
        print(f"📊 Final extracted data: {extracted}")
        return extracted

    def _calculate_confidence(self, extracted_data: Dict, doc_type: str) -> float:
        """Calculate confidence score based on extracted data"""
        if doc_type == 'unknown':
            return 0.3
        
        expected_fields = {
            'w2': ['wages', 'federal_tax'],
            '1099': ['income'],
            '1098': ['interest']
        }
        
        expected = expected_fields.get(doc_type, [])
        found = sum(1 for field in expected if field in extracted_data and extracted_data[field])
        
        if not expected:
            return 0.5
        
        confidence = found / len(expected)
        return min(0.95, max(0.3, confidence))

    def suggest_tax_entries(self, extracted_data: Dict) -> List[Dict]:
        """Generate tax entry suggestions based on extracted data"""
        suggestions = []
        
        # W-2 suggestions
        if 'wages' in extracted_data and extracted_data['wages']:
            suggestions.append({
                'field': 'income',
                'value': extracted_data['wages'],
                'description': f'W-2 Wages: ${extracted_data["wages"]:,.2f}',
                'confidence': 0.9
            })
        
        if 'federal_tax' in extracted_data and extracted_data['federal_tax']:
            suggestions.append({
                'field': 'withholdings',
                'value': extracted_data['federal_tax'],
                'description': f'Federal Tax Withheld: ${extracted_data["federal_tax"]:,.2f}',
                'confidence': 0.9
            })
        
        # 1099 suggestions
        if 'income' in extracted_data and extracted_data['income']:
            suggestions.append({
                'field': 'income',
                'value': extracted_data['income'],
                'description': f'1099 Income: ${extracted_data["income"]:,.2f}',
                'confidence': 0.8
            })
        
        return suggestions
