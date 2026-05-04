import numpy as np

class Adaline:
    def __init__(self, taxa_aprendizado=0.01, epocas=10):
        self.taxa_aprendizado = taxa_aprendizado
        self.epocas = epocas
        self.pesos = None
        self.bias = None

    def fit(self, x ,y):
        num_amostras, num_caracteristicas = x.shape

        self.pesos = np.zeros(num_caracteristicas)
        self.bias = 0

        for epoca in range(self.epocas):
            for idx_amostra, amostra in enumerate(x):
                saida_linear = np.dot(amostra, self.pesos) + self.bias

                erro = y[idx_amostra] - saida_linear

                atualizacao = self.taxa_aprendizado * erro
                self.pesos += atualizacao * amostra
                self.bias += atualizacao

    def predict(self, x):
        saida_linear = np.dot(x, self.pesos) + self.bias
        return saida_linear

    def funcao_ativacao(self, x):
        return np.where(x >= 0, 1, 0)

if __name__ == "__main__":
    x_treino = np.array([
        [0, 0], 
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    
    y_treino = np.array([0, 0, 0, 1])

    adaline = Adaline(taxa_aprendizado=0.01, epocas=10)
    adaline.fit(x_treino, y_treino)
    previsoes = adaline.funcao_ativacao(adaline.predict(x_treino))

    print("Previsões:", previsoes)
    print("Pesos:", adaline.pesos)
    print("Bias:", adaline.bias)
    print("Acurácia:", np.mean(previsoes == y_treino))