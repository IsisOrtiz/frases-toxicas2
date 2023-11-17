from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

# Função para carregar o conteúdo do arquivo
def carregar_arquivo(nome_arquivo):
    with open(nome_arquivo, 'r', encoding='utf-8') as arquivo:
        return arquivo.read().splitlines()

# Carregar palavras tóxicas do arquivo
palavras_toxicas = carregar_arquivo('arquivo.txt')

# Carregar frases que você quer verificar
frases = [
    "Seu animal",
    "Esta é uma frase de exemplo contendo a palavra anão.",
    "A palavra arrombado não deveria ser usada.",
    "Esta frase não contém palavras tóxicas.",
    "A palavra insignificante não deveria ser usada.",
    "Você é um amor",
    "A palavra sofrido não deveria ser usada.",
    "A palavra sofrimento não deveria ser usada.",
    "A palavra castigo não deveria ser usada.",
    "Sua mãe é uma vaca",
    ]

# Treinar o modelo Word2Vec
model = Word2Vec([frase.split() for frase in frases], min_count=1, vector_size=10)

# Função para calcular o vetor médio de uma frase
def calcular_vetor_medio(frase, model):
    palavras = frase.split()
    vetor_palavras = [model.wv.get_vector(palavra) for palavra in palavras if palavra in model.wv]
    if vetor_palavras:
        return sum(vetor_palavras) / len(vetor_palavras)
    else:
        return None

# Calcular vetores médios das palavras tóxicas
vetores_toxicos = [calcular_vetor_medio(palavra, model) for palavra in palavras_toxicas]

# Vetorizar as frases
vetores_frases = [calcular_vetor_medio(frase, model) for frase in frases]

# Função para calcular a similaridade de cosseno entre dois vetores
def calcular_similaridade(vetor1, vetor2):
    if vetor1 is not None and vetor2 is not None:
        return cosine_similarity([vetor1], [vetor2])[0][0]
    else:
        return 0.0

# Exibir cabeçalho
print("Frase Original | Frase Tóxica | Similaridade")

# Taxa de similaridade
taxa_similaridade = 0.1
# Marcar as frases como tóxicas
for i, vetor_frase in enumerate(vetores_frases):
    for j, vetor_toxico in enumerate(vetores_toxicos):
        similaridade = calcular_similaridade(vetor_frase, vetor_toxico)
        if similaridade > taxa_similaridade:  # Ajuste o limite conforme necessário
            print(f"{frases[i]} | {palavras_toxicas[j]} | {similaridade:.2f}")
