from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Função para carregar o conteúdo do arquivo
def carregar_arquivo(nome_arquivo):
    with open(nome_arquivo, 'r', encoding='utf-8') as arquivo:
        return arquivo.read().splitlines()

# Carregar palavras tóxicas do arquivo
palavras_toxicas = carregar_arquivo('arquivo.txt')

# Carregar frases que você quer verificar
frases = [
    "Esta é uma frase de exemplo contendo a palavra anão.",
    "A palavra arrombado não deveria ser usada.",
    "Esta frase não contém palavras tóxicas.",
    "A palavra insignificante não deveria ser usada.",
    "Você é um amor",
    "A palavra sofrido não deveria ser usada.",
]

# Treinar o modelo Word2Vec
model = Word2Vec([palavras_toxicas], min_count=1, vector_size=10)

# Vetorizar as palavras tóxicas
vectorizer = CountVectorizer().fit(palavras_toxicas)
toxic_vectors = vectorizer.transform(palavras_toxicas).toarray()

# Função para marcar as frases como tóxicas
def marcar_frases_toxicas(frases, model, vectorizer, toxic_vectors, threshold=0.8):
    frases_toxicas = []

    for frase in frases:
        # Vetorizar a frase
        frase_vector = vectorizer.transform([frase]).toarray()

        # Calcular similaridade de cosseno entre a frase e palavras tóxicas
        similarity = cosine_similarity(frase_vector, toxic_vectors).max()

        # Marcar a frase como tóxica se a similaridade for maior que o limite
        if similarity > threshold:
            frases_toxicas.append((frase, similarity))

    return frases_toxicas

# Marcar as frases como tóxicas
frases_toxicas = marcar_frases_toxicas(frases, model, vectorizer, toxic_vectors, threshold=0.5)

# Exibir frases tóxicas
print("Frases tóxicas:")
for frase, similarity in frases_toxicas:
    print(f"- '{frase}' (Similaridade: {similarity:.2f})")
