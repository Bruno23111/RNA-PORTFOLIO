import numpy as np

class Kohonen:
    def __init__(self, num_neuronios, taxa_aprendizado=0.1, epocas=10):
        self.num_neuronios = num_neuronios
        self.taxa_aprendizado = taxa_aprendizado
        self.epocas = epocas
        self.pesos = None

    def fit(self, x):
        num_amostras, num_caracteristicas = x.shape
        self.pesos = np.random.rand(self.num_neuronios, num_caracteristicas)

        for epoca in range(self.epocas):
            for amostra in x:
                distancias = np.linalg.norm(self.pesos - amostra, axis=1)
                indice_vencedor = np.argmin(distancias)
                self.pesos[indice_vencedor] += self.taxa_aprendizado * (amostra - self.pesos[indice_vencedor])

    def predict(self, x):
        distancias = np.linalg.norm(self.pesos - x, axis=1)
        return np.argmin(distancias)

if __name__ == "__main__":
    x_treino = np.array([
        [0.1, 0.2],
        [0.2, 0.1],
        [0.8, 0.9],
        [0.9, 0.8]
    ])

    kohonen = Kohonen(num_neuronios=2, taxa_aprendizado=0.1, epocas=10)
    kohonen.fit(x_treino)

    for amostra in x_treino:
        indice_vencedor = kohonen.predict(amostra)
        print(f"Amostra: {amostra}, Neurônio Vencedor: {indice_vencedor}")