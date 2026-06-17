# raghelper.py
from google.genai import types

class RagBase:
    def __init__(self, index, client, model="gemini-2.5-flash", course="llm-zoomcamp"):
        # Encapsulating dependencies so they aren't bound to global notebook variables
        self.index = index
        self.client = client
        self.model = model
        self.course = course
        
        # Default templates
        self.instructions = """
        Your task is to answer questions from the course participants based on the provided context.
        Use the context to find relevant information and provide accurate answers. 
        If the answer is not found in the context, respond with "I don't know."
        """
        self.user_prompt_template = "Question:\n{question}\n\nContext:\n{context}"

    def search(self, query):
        return self.index.search(
            query,
            boost_dict={"question": 2.0, "section": 0.5},
            filter_dict={"course": self.course},
            num_results=5
        )

    def build_context(self, search_results):
        lines = []
        for doc in search_results:
            lines.append(doc["section"])
            lines.append("Q: " + doc["question"])
            lines.append("A: " + doc["answer"])
            lines.append("")
        return "\n".join(lines).strip()

    def build_prompt(self, question, search_results):
        context = self.build_context(search_results)
        prompt = self.user_prompt_template.format(
            question=question,
            context=context
        )
        return prompt.strip()

    def llm(self, prompt):
        config = types.GenerateContentConfig(
            system_instruction=self.instructions
        )
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config
        )
        return response.text

    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        answer = self.llm(prompt)
        return answer