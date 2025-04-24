INSTRUCTION_CORP_FIN = """
I will give you a question and a document.

You need to answer the question based on the document.

--QUESTION--
{question}
--END OF QUESTION--

--DOCUMENT--
{document}
--END OF DOCUMENT--

Your answer:
"""

INSTRUCTION_EDGAR_RESEARCH = """
I will give you a question and a set of documents.

You need to answer the question based on the documents.

--QUESTION--
{question}
--END OF QUESTION--

--DOCUMENTS--
{documents}
--END OF DOCUMENTS--

Your answer:
"""