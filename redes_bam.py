import numpy as np

class RedeBAM:
    def __init__(self, num_x, num_y):
        self.pesos = np.zeros((num_x, num_y))

    def treinar(self, pares):
        for x, y in pares:
            self.pesos += np.outer(x, y)

    def recuperar_frente(self, x):
        y = np.sign(np.dot(x, self.pesos))
        y[y == 0] = 1
        return y

    def recuperar_tras(self, y):
        x = np.sign(np.dot(y, self.pesos.T))
        x[x == 0] = 1
        return x

    def recuperar(self, x_inicial, max_iter=10):
        x_atual = np.copy(x_inicial)
        
        for _ in range(max_iter):
            y_atual = self.recuperar_frente(x_atual)
            x_novo = self.recuperar_tras(y_atual)
            
            if np.array_equal(x_atual, x_novo):
                break
                
            x_atual = x_novo
            
        return x_atual, y_atual

if __name__ == "__main__":
    bam = RedeBAM(num_x=4, num_y=2)
    
    pares_treino = [
        (np.array([1, 1, -1, -1]), np.array([1, -1])),
        (np.array([-1, -1, 1, 1]), np.array([-1, 1]))
    ]
    
    bam.treinar(pares_treino)
    
    x_teste = np.array([1, 1, 1, -1])
    x_recuperado, y_recuperado = bam.recuperar(x_teste)
    
    print("X Inicial:", x_teste)
    print("X Recuperado:", x_recuperado)
    print("Y Recuperado:", y_recuperado)