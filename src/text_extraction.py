import PyPDF2

def textExtraction(pdf_files):    
    pdf_texts = []
    for pdf_file in pdf_files:
        reader = PyPDF2.PdfReader(pdf_file)
        pdf_text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            pdf_text += page.extract_text()
        pdf_texts.append(pdf_text)

    txt = "\n".join(pdf_texts)
    
    return txt