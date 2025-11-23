def clean_ocr_text(text: str) -> str:
    replacements = {
        'о': 'о', 'а': 'а', 'е': 'е', 'р': 'р', 'с': 'с',
        'О': 'О', 'А': 'А', 'Е': 'Е',
        'ё': 'ё', 'Ё': 'Ё',
        '«': '"', '»': '"', '“': '"', '”': '"',
        '−': '-', '–': '-', '—': '-',
        '  ': ' ', '\t': ' ',
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text.strip()
