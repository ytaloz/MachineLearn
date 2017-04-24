# Atividade retirada do curso de Machine Learn da Alura
# O objetivo dessa atividade e classificar dois tipos de animais "Porco" e "Cachorro"
# Dados de entrada para treinamento 
# Utilizaremos todos os dados durante o treinamento para teste

porco1 = [1, 1, 0]
porco2 = [1, 1, 0]
porco3 = [1, 1, 0]

cachorro4 = [1, 1, 1]
cachorro5 = [0, 1, 1]
cachorro6 = [0, 1, 1]

# Conjunto de dados 
dados = [porco1, porco2, porco3, cachorro4, cachorro5, cachorro6]
marcacoes = [1, 1, 1, -1, -1, -1]

# Biblioteca python sklearn
from sklearn.naive_bayes import MultinomialNB

# Criar um modelo seguindo o naive_bayes
modelo = MultinomialNB()

# Realiza o treino do adequa o modelo
modelo.fit(dados, marcacoes)

# Dados que qeremos classificar
misterioso1 = [1, 1, 1]
misterioso2 = [1, 0, 0]
misterioso3 = [0, 0, 1]

teste = [misterioso1, misterioso2, misterioso3]

# Realizamos a identificacao dos animais "Predição"
resultado = modelo.predict(teste)

print(resultado)