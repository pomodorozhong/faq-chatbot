import ollama
import chromadb
import chromadb.utils.embedding_functions as embedding_functions

instructor_collection = None

def test_ollama():
    response = ollama.generate(model='llama3.1', prompt='Hi')
    print(response['response'])

def setup_memory():
    global instructor_collection
    client = chromadb.Client()

    ollama_ef = embedding_functions.OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text",
    )

    instructor_collection = client.create_collection(
        name="faq_embeddings", embedding_function=ollama_ef)

    faq_documents = [
        "糖蘋果工作室(Sugar Apple Studio)的服務範圍包含: 原生 MacOS/iOS 開發、Python 開發、客製化 AI Agent、 AI 聊天機器人、樹莓派相關應用、電腦視覺/機器視覺應用",
        "糖蘋果工作室(Sugar Apple Studio)由兩個人組成."
    ]

    instructor_collection.add(
        documents=faq_documents,
        ids=[str(i) for i in range(len(faq_documents))]
    )

def ask_questions_given_context(query) -> str:
    global instructor_collection
    relevant_docs = instructor_collection.query(
        query_texts=[query],
        n_results=1
    )

    response = ollama.generate(model='llama3.1',
                               prompt=f'answer the question: ```{query}```, considering the following context: ```{relevant_docs}```. Keep the answer concise and under 10 words.'
                               )
    return response['response']

def ask_questions():
    global instructor_collection
    questions = [
        "糖蘋果工作室的服務有哪些?",
        "糖蘋果工作室的成員有幾位?",
        "How many members are in the Sugar Apple Studio?",
    ]

    if instructor_collection is None:
        for i in range(len(questions)):
            print(f'\nQ{i+1}: {questions[i]}')
            query = questions[i]
            response = ollama.generate(model='llama3.1',
                                    prompt=f'answer the question: ```{query}```. Keep the answer concise and under 10 words.'
                                    )
            print(f'A{i+1}: ' + response['response'])
        print('\n\n')
    else:
        for i in range(len(questions)):
            print(f'\nQ{i+1}: {questions[i]}')
            response = ask_questions_given_context(questions[i])
            print(f'A{i+1}: ' + response)

        print('\n\n')
        print('\n\n')
        print('\n\n')
        print('\n\n')
        print('\n\n')
        print('\n\n')
        print('\n\n')
        print('\n\n')
        print('\n\n')
        print('\n\n')
        print('\n\n')
        print('\n\n')
        print('\n\n')
        print('\n\n')

if __name__ == "__main__":
    print('1️⃣ 測試 LLM 基礎知識')
    print('====================')
    ask_questions()

    setup_memory()
    print('2️⃣ 新增記憶後，再次測試 LLM 知識')
    print('====================')
    ask_questions()

