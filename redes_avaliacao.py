import numpy as np
from perceptron_simples import PerceptronSimples
from adaline import Adaline
from redes_avaliacao import AvaliacaoRedesNeurais

class AvaliacaoRedesNeurais:
    def __init__(self):
        pass

    def avaliar(self, modelo, x_teste, y_teste):
        previsoes = modelo.predict(x_teste)
        
        # 1. Garante que as previsões sejam um array 1D
        if isinstance(previsoes, np.ndarray):
            if previsoes.ndim > 1:
                previsoes = previsoes.flatten()
            
            # Se as previsões tiverem casas decimais (saída do MLP com sigmoide),
            # aplicamos o degrau de 0.5. Se já forem inteiros (Perceptron), não afeta.
            if previsoes.dtype == float or previsoes.dtype == np.float64:
                previsoes = (previsoes > 0.5).astype(int)

        # 2. Garante que o y_teste também seja 1D para evitar erros de matriz
        y_teste_1d = np.array(y_teste).flatten()

        # 3. Calcula a acurácia com segurança
        acuracia = np.mean(previsoes == y_teste_1d)
        return acuracia

if __name__ == "__main__":
    from perceptron_simples import PerceptronSimples
    from adaline import Adaline
    from multilayer_perceptron import MultilayerPerceptron

    # O ideal é separar treino e teste, mas manteremos seu exemplo de porta lógica AND
    x_dados = np.array([
        [0, 0], 
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y_dados = np.array([0, 0, 0, 1])

    avaliacao = AvaliacaoRedesNeurais()

    print("--- Resultados da Avaliação ---")

    # 1. Perceptron Simples
    perceptron = PerceptronSimples(taxa_aprendizado=0.1, epocas=10)
    perceptron.fit(x_dados, y_dados)
    acuracia_perceptron = avaliacao.avaliar(perceptron, x_dados, y_dados)
    print(f"Acurácia do Perceptron: {acuracia_perceptron * 100:.2f}%")

    # 2. Adaline
    adaline = Adaline(taxa_aprendizado=0.01, epocas=50) # Adaline costuma precisar de mais épocas
    adaline.fit(x_dados, y_dados)
    acuracia_adaline = avaliacao.avaliar(adaline, x_dados, y_dados)
    print(f"Acurácia do Adaline: {acuracia_adaline * 100:.2f}%")

    # 3. Multilayer Perceptron
    mlp = MultilayerPerceptron(taxa_aprendizado=0.1, epocas=10000)
    # MLP costuma exigir o y no formato (N, 1) durante o treino
    mlp.fit(x_dados, y_dados.reshape(-1, 1)) 
    acuracia_mlp = avaliacao.avaliar(mlp, x_dados, y_dados)
    print(f"Acurácia do MLP: {acuracia_mlp * 100:.2f}%")