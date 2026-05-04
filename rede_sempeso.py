import numpy as np

class RedeSemPeso:
    def __init__(self):
        self.ram = {}

    def fit(self, x, y):
        for amostra, rotulo in zip(x, y):
            endereco = tuple(amostra)
            if rotulo not in self.ram:
                self.ram[rotulo] = set()
            self.ram[rotulo].add(endereco)

    def predict(self, x):
        previsoes = []
        for amostra in x:
            endereco = tuple(amostra)
            classe_prevista = 0
            for classe, enderecos in self.ram.items():
                if endereco in enderecos:
                    classe_prevista = classe
                    break
            previsoes.append(classe_prevista)
        return np.array(previsoes)

if __name__ == "__main__":
    x_treino = np.array([
        [0, 0], 
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    
    y_treino = np.array([0, 0, 0, 1])

    rede = RedeSemPeso()
    rede.fit(x_treino, y_treino)
    previsoes = rede.predict(x_treino)

    print("Previsões:", previsoes)
    print("Conteúdo da RAM:", rede.ram)
    print("Acurácia:", np.mean(previsoes == y_treino))