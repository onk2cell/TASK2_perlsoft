import re
from rag_engine import results_dict

qa_dict = results_dict

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = text.split()
    return tokens

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_most_similar_answer(new_question, qa_dict):
    new_question = preprocess(new_question)
    tfidf_vectorizer = TfidfVectorizer()
    question_vectors = tfidf_vectorizer.fit_transform(qa_dict.keys())
    new_question_vector = tfidf_vectorizer.transform([" ".join(new_question)])

    similarities = cosine_similarity(new_question_vector, question_vectors)
    most_similar_index = similarities.argmax()
    most_similar_question = list(qa_dict.keys())[most_similar_index]
    most_similar_answer = qa_dict[most_similar_question]

    return most_similar_answer


new_question = " ?"
most_similar_answer = get_most_similar_answer(new_question, qa_dict)
print("Answer:", most_similar_answer)