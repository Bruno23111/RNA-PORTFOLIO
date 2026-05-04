import numpy as np

class PerceptronSimples:
    #INICIA PERCEPTRON SIMPLES COM TAXA DE APRENDIZADO E ÉPOCAS
    def __init__(self, taxa_aprendizado=0.1, epocas=10):
        self.taxa_aprendizado = taxa_aprendizado
        self.epocas = epocas
        self.pesos = None
        self.bias = None

    #AJUSTA OS PESOS E O BIAS COM BASE NAS AMOSTRAS DE TREINAMENTO
    def fit(self, x ,y):
        num_amostras, num_caracteristicas = x.shape

        self.pesos = np.zeros(num_caracteristicas)
        self.bias = 0

        for epoca in range(self.epocas):
            for idx_amostra, amostra in enumerate(x):
                saida_linear = np.dot(amostra, self.pesos) + self.bias
                y_previsto = self.funcao_ativacao(saida_linear)

                erro = y[idx_amostra] - y_previsto

                atualizacao = self.taxa_aprendizado * erro
                self.pesos += atualizacao * amostra
                self.bias += atualizacao

    #FUNÇÃO DE ATIVAÇÃO
    def funcao_ativacao(self, x):
        return np.where(x >= 0, 1, 0)


    def predict(self, x):
        saida_linear = np.dot(x, self.pesos) + self.bias
        y_previsto = self.funcao_ativacao(saida_linear)
        return y_previsto

if __name__ == "__main__":
    x_treino = np.array([
        [0, 0], 
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    
    y_treino = np.array([0, 0, 0, 1])

    perceptron = PerceptronSimples(taxa_aprendizado=0.1, epocas=10)
    perceptron.fit(x_treino, y_treino)
    previsoes = perceptron.predict(x_treino)

    print("Previsões:", previsoes)
    print("Pesos:", perceptron.pesos)
    print("Bias:", perceptron.bias)
    print("Acurácia:", np.mean(previsoes == y_treino))
