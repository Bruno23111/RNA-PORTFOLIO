import numpy as np

class Hopfield:
    def __init__(self, num_neuronios):
        self.num_neuronios = num_neuronios
        self.pesos = np.zeros((num_neuronios, num_neuronios))

    def treinar(self, padroes):
        for padrao in padroes:
            self.pesos += np.outer(padrao, padrao)
        np.fill_diagonal(self.pesos, 0)

    def recuperar(self, padrao_inicial, max_iter=100):
        estado_atual = np.copy(padrao_inicial)
        
        for _ in range(max_iter):
            estado_anterior = np.copy(estado_atual)
            
            for i in range(self.num_neuronios):
                soma = np.dot(self.pesos[i], estado_atual)
                estado_atual[i] = 1 if soma >= 0 else -1
                
            if np.array_equal(estado_atual, estado_anterior):
                break
                
        return estado_atual

if __name__ == "__main__":
    hopfield = Hopfield(num_neuronios=4)
    
    padroes_treino = np.array([
        [1, -1, 1, -1],
        [-1, 1, -1, 1]
    ])
    
    hopfield.treinar(padroes_treino)
    
    padrao_inicial = np.array([1, -1, -1, -1])
    padrao_recuperado = hopfield.recuperar(padrao_inicial)
    
    print("Padrão Inicial:", padrao_inicial)
    print("Padrão Recuperado:", padrao_recuperado)