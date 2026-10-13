from langchain_openai import ChatOpenAI

class AnswerGenerator:
    """Generate answers using retrieved document context."""

    def __init__(self, 
                 model_name : str = "gpt-4o-mini",
                 temperature: float = 0.0
                 ) -> None:
        self.llm = ChatOpenAI(
            model=model_name, 
            temperature=temperature
            )


    def generate(self, question: str, context: str) -> str:
        if not question.strip():
            raise ValueError("Question cannot be empty")

        if not context.strip():
            raise ValueError("Context cannot be empty")

        RAG_PROMPT = f"""
                You are an document question and answering assitant
                Answer the user questions using ONLY the provided document context.

                If the answer cannot be foumd in the context.
                say:

                'I donot have enough information provided in the documents 
                to answer this question.'

                Do not invent information.

                Document context:
                ---------------------
                {context}
                ---------------------

                User Question:
                {question}

                Answer:

                """

        response = self.llm.invoke(RAG_PROMPT)

        return response.content

        

