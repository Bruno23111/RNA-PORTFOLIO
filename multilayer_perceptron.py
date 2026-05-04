import numpy as np

class MultilayerPerceptron:
    def __init__(self, taxa_aprendizado=0.1, epocas=10000):
        self.taxa_aprendizado = taxa_aprendizado
        self.epocas = epocas
        self.pesos_entrada_oculta = None
        self.pesos_oculta_saida = None
        self.bias_oculta = None
        self.bias_saida = None

    def funcao_ativacao(self, x):
        # Sigmoide: curva suave entre 0 e 1
        return 1 / (1 + np.exp(-x))

    def funcao_derivada(self, x):
        # Derivada da Sigmoide: f(x) * (1 - f(x))
        # Note: aqui 'x' já deve ser o valor que passou pela sigmoide
        return x * (1 - x)

    def fit(self, x, y):
        num_amostras, num_caracteristicas = x.shape
        num_neuronios_oculta = 4 

        # Inicialização Aleatória (Essencial para MLP)
        self.pesos_entrada_oculta = np.random.uniform(-1, 1, (num_caracteristicas, num_neuronios_oculta))
        self.pesos_oculta_saida = np.random.uniform(-1, 1, (num_neuronios_oculta, 1))
        self.bias_oculta = np.random.uniform(-1, 1, (1, num_neuronios_oculta))
        self.bias_saida = np.random.uniform(-1, 1, (1, 1))

        for epoca in range(self.epocas):
            # Forward Pass (Processamento)
            camada_oculta_input = np.dot(x, self.pesos_entrada_oculta) + self.bias_oculta
            camada_oculta_output = self.funcao_ativacao(camada_oculta_input)

            camada_saida_input = np.dot(camada_oculta_output, self.pesos_oculta_saida) + self.bias_saida
            camada_saida_output = self.funcao_ativacao(camada_saida_input)

            # Backpropagation (Cálculo do Erro e Gradiente)
            erro_saida = y - camada_saida_output
            d_saida = erro_saida * self.funcao_derivada(camada_saida_output)

            erro_camada_oculta = d_saida.dot(self.pesos_oculta_saida.T)
            d_oculta = erro_camada_oculta * self.funcao_derivada(camada_oculta_output)

            # Atualização de Pesos e Bias
            self.pesos_oculta_saida += camada_oculta_output.T.dot(d_saida) * self.taxa_aprendizado
            self.bias_saida += np.sum(d_saida, axis=0, keepdims=True) * self.taxa_aprendizado
            
            self.pesos_entrada_oculta += x.T.dot(d_oculta) * self.taxa_aprendizado
            self.bias_oculta += np.sum(d_oculta, axis=0, keepdims=True) * self.taxa_aprendizado

    def predict(self, x):
        camada_oculta_input = np.dot(x, self.pesos_entrada_oculta) + self.bias_oculta
        camada_oculta_output = self.funcao_ativacao(camada_oculta_input)
        camada_saida_input = np.dot(camada_oculta_output, self.pesos_oculta_saida) + self.bias_saida
        return self.funcao_ativacao(camada_saida_input)

if __name__ == "__main__":
    # Testando com XOR (O desafio que o Perceptron não resolve)
    x_treino = np.array([[0,0], [0,1], [1,0], [1,1]])
    y_treino = np.array([[0], [1], [1], [0]]) 

    mlp = MultilayerPerceptron(taxa_aprendizado=0.2, epocas=20000)
    mlp.fit(x_treino, y_treino)
    previsoes = mlp.predict(x_treino)

    print("--- Resultados MLP (XOR) ---")
    for i in range(len(x_treino)):
        print(f"Entrada: {x_treino[i]} -> Saída: {previsoes[i][0]:.4f} (Alvo: {y_treino[i][0]})")