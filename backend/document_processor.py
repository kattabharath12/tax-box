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
