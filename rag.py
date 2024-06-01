
class DocumentProcessor:
    def __init__(self, data_path, chroma_path, chunk_size=400, chunk_overlap=100):
        self._data_path = data_path
        self._chroma_path = chroma_path
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._embedding_function = OpenAIEmbeddings()
        self._prompt_template = """
        Answer the question based only on the following context:

        {context}

        ---

        Answer the question based on the above context: {question} 
        Make answer in Introduction, then body and then conclusion. 
        The whole answer will be in 250-300 words. 
        """
        self._load_documents()
        self._create_vector_store()

    def _load_documents(self):
        print("Loading documents...")
        document_loader = PyPDFDirectoryLoader(self._data_path)
        self.documents = document_loader.load()
        print(f"Loaded {len(self.documents)} documents.")

    def _split_text(self):
        print("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            length_function=len,
            add_start_index=True,
        )
        self.chunks = text_splitter.split_documents(self.documents)
        print(f"Split {len(self.documents)} documents into {len(self.chunks)} chunks.")

    def _create_vector_store(self):
        self._split_text()
        print("Creating Chroma vector store...")
        if os.path.exists(self._chroma_path):
            shutil.rmtree(self._chroma_path)
        
        self.vector_store = Chroma.from_documents(
            self.chunks, self._embedding_function, persist_directory=self._chroma_path
        )
        self.vector_store.persist()
        print(f"Saved chunks to {self._chroma_path}.")

    def _create_vector_store(self, collection_name='polity', clear_persist_folder: bool = True):
        self._split_text()
        if clear_persist_folder:
            pf = Path(self._chroma_path)
            if pf.exists() and pf.is_dir():
                print(f"Deleting the content of: {pf}")
                shutil.rmtree(pf)
            pf.mkdir(parents=True, exist_ok=True)
            print(f"Recreated the directory at: {pf}")
        print("Generating and persisting the embeddings..")
        self.vector_store = Chroma.from_documents(
            self.chunks, self._embedding_function, persist_directory=self._chroma_path
        )
        self.vector_store.persist()
        print(f"Saved chunks to {self._chroma_path}.")
        
    def generate_data_store(self):
        """
        Generate a vector database in Chroma from documents.
        """
        self._create_vector_store()

    def _generate_multi_query_prompt(self, question):
        template = """You are an AI language model assistant. Your task is to generate five 
        different versions of the given user question to retrieve relevant documents from a vector 
        database. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of the distance-based similarity search. 
        Provide these alternative questions separated by newlines. Original question: {question}"""
        prompt_perspectives = ChatPromptTemplate.from_template(template)
        return prompt_perspectives.format(question=question)

    def _generate_decomposition_prompt(self, question):
        template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
        The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
        Generate multiple search queries related to: {question} \n
        Output (3 queries):"""
        prompt_decomposition = ChatPromptTemplate.from_template(template)
        return prompt_decomposition.format(question=question)

    def _format_qa_pair(self, question, answer):
        return f"Question: {question}\nAnswer: {answer}\n\n"

    def _generate_queries(self, question, prompt_template):
        model = ChatOpenAI(temperature=0)
        output_parser = StrOutputParser()
        prompt = ChatPromptTemplate.from_template(prompt_template)
        response = LLMChain(model=model, prompt=prompt, output_parser=output_parser).invoke({"question": question})
        return response.split("\n")

    def _retrieve_documents(self, queries):
        documents = [self.vector_store.similarity_search(query) for query in queries]
        unique_docs = self._get_unique_union(documents)
        return unique_docs

    def _get_unique_union(self, documents):
        flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
        unique_docs = list(set(flattened_docs))
        return [loads(doc) for doc in unique_docs]

    def _answer_recursively(self, question):
        decomposition_prompt = self._generate_decomposition_prompt(question)
        sub_questions = self._generate_queries(question, decomposition_prompt)
        q_a_pairs = ""

        for sub_question in sub_questions:
            retrieved_docs = self.vector_store.similarity_search(sub_question)
            context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
            answer_prompt = ChatPromptTemplate.from_template(self._prompt_template).format(context=context, question=sub_question)
            model = ChatOpenAI(temperature=0)
            answer = LLMChain(model=model, prompt=ChatPromptTemplate.from_template(answer_prompt), output_parser=StrOutputParser()).invoke({})
            q_a_pair = self._format_qa_pair(sub_question, answer)
            q_a_pairs += f"\n---\n{q_a_pair}"

        return q_a_pairs

    def _rag_fusion(self, question):
        prompt_rag_fusion = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
        Generate multiple search queries related to: {question} \n
        Output (4 queries):"""
        queries = self._generate_queries(question, prompt_rag_fusion)
        retrieved_docs = [self.vector_store.similarity_search(query) for query in queries]
        final_docs = self._reciprocal_rank_fusion(retrieved_docs)
        return final_docs

    def _reciprocal_rank_fusion(self, results, k=60):
        fused_scores = {}
        for docs in results:
            for rank, doc in enumerate(docs):
                doc_str = dumps(doc)
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                fused_scores[doc_str] += 1 / (rank + k)

        reranked_results = [
            (loads(doc), score) for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        ]
        return [doc for doc, _ in reranked_results]

    def _final_rag_chain(self, question, context):
        template = """Answer the following question based on this context:

        {context}

        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template).format(context=context, question=question)
        model = ChatOpenAI(temperature=0)
        response = LLMChain(model=model, prompt=ChatPromptTemplate.from_template(prompt), output_parser=StrOutputParser()).invoke({})
        return response

    def search_and_respond(self, query_text):
        print(f"Processing query: {query_text}")
        multi_query_prompt = self._generate_multi_query_prompt(query_text)
        queries = self._generate_queries(query_text, multi_query_prompt)
        retrieved_docs = self._retrieve_documents(queries)
        context_text = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
        response = self._final_rag_chain(query_text, context_text)
        print("Response generated successfully.")
        return response

    def search_and_respond_recursively(self, query_text):
        print(f"Processing query recursively: {query_text}")
        q_a_pairs = self._answer_recursively(query_text)
        context_text = q_a_pairs
        response = self._final_rag_chain(query_text, context_text)
        print("Recursive response generated successfully.")
        return response

    def search_and_respond_rag_fusion(self, query_text):
        print(f"Processing query with RAG Fusion: {query_text}")
        final_docs = self._rag_fusion(query_text)
        context_text = "\n\n---\n\n".join([doc.page_content for doc in final_docs])
        response = self._final_rag_chain(query_text, context_text)
        print("RAG Fusion response generated successfully.")
        return response

