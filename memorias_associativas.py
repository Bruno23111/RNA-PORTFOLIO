import numpy as np

class MemoriaAssociativaHopfield:
    def __init__(self, num_caracteristicas):
        self.pesos = np.zeros((num_caracteristicas, num_caracteristicas))

    def armazenar(self, padroes):
        """
        Treina a rede usando a Regra de Hebb.
        Os padrões são 'memorizados' na matriz de pesos.
        """
        for padrao in padroes:
            p = padrao.reshape(-1, 1)
            self.pesos += np.dot(p, p.T)
        
        np.fill_diagonal(self.pesos, 0)

    def recuperar(self, padrao_entrada, passos=3):
        """
        Tenta recuperar uma memória a partir de uma entrada (que pode ter ruído).
        """
        estado_atual = padrao_entrada.copy()
        
        for _ in range(passos):
            soma = np.dot(self.pesos, estado_atual)
            estado_atual = np.sign(soma)
            estado_atual[estado_atual == 0] = 1
            
        return estado_atual

if __name__ == "__main__":
    padrao_A = np.array([ 1,  1, -1, -1])
    padrao_B = np.array([-1, -1,  1,  1])
    
    memorias_para_aprender = [padrao_A, padrao_B]

    memoria_ia = MemoriaAssociativaHopfield(num_caracteristicas=4)
    memoria_ia.armazenar(memorias_para_aprender)
    
    print("Pesos aprendidos pela rede:\n", memoria_ia.pesos)
    print("-" * 30)
    
    padrao_com_ruido = np.array([ 1, -1, -1, -1]) 
    
    print(f"Padrão Original A: {padrao_A}")
    print(f"Padrão com Ruído:  {padrao_com_ruido}")
    
    padrao_recuperado = memoria_ia.recuperar(padrao_com_ruido)
    
    print(f"Padrão Recuperado: {padrao_recuperado}")
    
    if np.array_equal(padrao_recuperado, padrao_A):
        print("Sucesso! A rede filtrou o ruído e lembrou do Padrão A original.")