import numpy as np

class RedeConvolucional:
    def __init__(self, num_filtros, tamanho_filtro):
        self.num_filtros = num_filtros
        self.tamanho_filtro = tamanho_filtro
        self.filtros = np.random.randn(num_filtros, tamanho_filtro, tamanho_filtro)

    def convolucao(self, imagem):
        altura_imagem, largura_imagem = imagem.shape
        altura_saida = altura_imagem - self.tamanho_filtro + 1
        largura_saida = largura_imagem - self.tamanho_filtro + 1
        saida = np.zeros((self.num_filtros, altura_saida, largura_saida))

        for f in range(self.num_filtros):
            for i in range(altura_saida):
                for j in range(largura_saida):
                    regiao = imagem[i:i+self.tamanho_filtro, j:j+self.tamanho_filtro]
                    saida[f, i, j] = np.sum(regiao * self.filtros[f])
        
        return saida

if __name__ == "__main__":
    imagem_teste = np.array([
        [1, 2, 3, 0],
        [0, 1, 2, 3],
        [3, 0, 1, 2],
        [2, 3, 0, 1]
    ])

    rede = RedeConvolucional(num_filtros=2, tamanho_filtro=2)
    resultado_convolucao = rede.convolucao(imagem_teste)

    print("Resultado da Convolução:")
    print(resultado_convolucao)