# raghelper.py
from google.genai import types

class RagBase:
    # 1. Update the initialization parameters to take es_client instead of a minsearch index
    def __init__(self, es_client, index_name="course-questions", client=None, model="gemini-2.5-flash", course="llm-zoomcamp"):
        self.es_client = es_client
        self.index_name = index_name
        self.client = client
        self.model = model
        self.course = course
        
        self.instructions = """
        Your task is to answer questions from the course participants based on the provided context.
        Use the context to find relevant information and provide accurate answers. 
        If the answer is not found in the context, respond with "I don't know."
        """
        self.user_prompt_template = "Question:\n{question}\n\nContext:\n{context}"

    # 2. Rewrite the search method using your working Elasticsearch DSL query
    def search(self, query):
        search_query = {
            "size": 5,
            "query": {
                "bool": {
                    "must": {
                        "multi_match": {
                            "query": query,
                            "fields": ["question^2", "text", "section"],
                            "type": "best_fields"
                        }
                    },
                    "filter": {
                        "term": {
                            "course": self.course
                        }
                    }
                }
            }
        }
        
        response = self.es_client.search(index=self.index_name, body=search_query)
        
        # 3. Cleanly parse the nested results list before passing it down the pipeline
        result_docs = []
        for hit in response['hits']['hits']:
            result_docs.append(hit['_source'])
            
        return result_docs

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