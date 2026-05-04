import numpy as np

class RedesArt:
    def __init__(self, vigilancia=0.8, taxa_aprendizado=0.5):
        self.vigilancia = vigilancia
        self.taxa_aprendizado = taxa_aprendizado
        self.pesos = []

    def fit(self, x):
        for amostra in x:
            if not self.pesos:
                self.pesos.append(np.copy(amostra))
                continue

            ativacoes = []
            for w in self.pesos:
                intersecao = np.minimum(amostra, w)
                ativacao = np.sum(intersecao) / (0.01 + np.sum(w))
                ativacoes.append(ativacao)

            indices_ordenados = np.argsort(ativacoes)[::-1]
            ressonancia_alcancada = False

            for idx in indices_ordenados:
                w = self.pesos[idx]
                intersecao = np.minimum(amostra, w)
                correspondencia = np.sum(intersecao) / np.sum(amostra)

                if correspondencia >= self.vigilancia:
                    self.pesos[idx] = self.taxa_aprendizado * intersecao + (1 - self.taxa_aprendizado) * w
                    ressonancia_alcancada = True
                    break

            if not ressonancia_alcancada:
                self.pesos.append(np.copy(amostra))

    def predict(self, x):
        previsoes = []
        for amostra in x:
            if not self.pesos:
                previsoes.append(-1)
                continue
                
            ativacoes = []
            for w in self.pesos:
                intersecao = np.minimum(amostra, w)
                ativacao = np.sum(intersecao) / (0.01 + np.sum(w))
                ativacoes.append(ativacao)
            previsoes.append(np.argmax(ativacoes))
        return np.array(previsoes)

if __name__ == "__main__":
    x_treino = np.array([
        [0.1, 0.2],
        [0.2, 0.1],
        [0.8, 0.9],
        [0.9, 0.8]
    ])

    redes_art = RedesArt(vigilancia=0.85, taxa_aprendizado=0.5)
    redes_art.fit(x_treino)

    for amostra in x_treino:
        indice_vencedor = redes_art.predict([amostra])[0]
        print(f"Amostra: {amostra}, Categoria: {indice_vencedor}")