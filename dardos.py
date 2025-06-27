import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

class JogoDeDardos:
    def __init__(self):
        self.A = None
        self.B = None
        self.C = None
        self.lancador = None
        self.num_lancamentos = 0
        self.vetores_direcao = []
        self.centro = None
        self.Am = None
        self.Bm = None
        self.Cm = None
        self.pontuacao = 0
        self.pontos_interseccao = []

    def ler_entrada(self):
        print("Digite as coordenadas dos pontos A, B, C (um por linha):")
        self.A = np.array([float(x) for x in input().split()])
        self.B = np.array([float(x) for x in input().split()])
        self.C = np.array([float(x) for x in input().split()])

        print("Digite as coordenadas do ponto de lançamento:")
        self.lancador = np.array([float(x) for x in input().split()])

        print("Digite o número de lançamentos:")
        self.num_lancamentos = int(input())

        print(f"Digite os {self.num_lancamentos} vetores direção (um por linha):")
        for _ in range(self.num_lancamentos):
            direcao = np.array([float(x) for x in input().split()])
            self.vetores_direcao.append(direcao)

    def validar_entrada(self):
        v1 = self.B - self.A
        v2 = self.C - self.A
        normal = np.cross(v1, v2)
        if np.linalg.norm(normal) < 1e-10:
            raise ValueError("Os pontos A, B e C não formam um triângulo válido.")

        d = -np.dot(normal, self.A)
        if abs(np.dot(normal, self.lancador) + d) < 1e-10:
            raise ValueError("O ponto de lançamento está no mesmo plano do triângulo.")

    def calcular_alvos(self):
        self.centro = (self.A + self.B + self.C) / 3
        self.Am = self.centro + 0.5 * (self.A - self.centro)
        self.Bm = self.centro + 0.5 * (self.B - self.centro)
        self.Cm = self.centro + 0.5 * (self.C - self.centro)

    def ponto_no_triangulo(self, ponto, P1, P2, P3):
        def sinal(a, b, c):
            return (a[0] - c[0])*(b[1] - c[1]) - (b[0] - c[0])*(a[1] - c[1])

        s1 = sinal(ponto, P1, P2) <= 0.0
        s2 = sinal(ponto, P2, P3) <= 0.0
        s3 = sinal(ponto, P3, P1) <= 0.0

        return (s1 == s2) and (s2 == s3)

    def intersecao_reta_plano(self, direcao):
        normal = np.cross(self.B - self.A, self.C - self.A)
        d = -np.dot(normal, self.A)

        denominador = np.dot(normal, direcao)

        if abs(denominador) < 1e-10:
            return None

        t = -(np.dot(normal, self.lancador) + d) / denominador

        if t < 0:
            return None

        ponto_intersecao = self.lancador + t * direcao
        return ponto_intersecao

    def jogar(self):
        self.calcular_alvos()

        for direcao in self.vetores_direcao:
            intersecao = self.intersecao_reta_plano(direcao)
            self.pontos_interseccao.append(intersecao)

            if intersecao is None:
                self.pontuacao -= 1
                continue

            if np.linalg.norm(intersecao - self.centro) < 1e-5:
                self.pontuacao += 20
                continue

            if self.ponto_no_triangulo(intersecao[:2], self.Am[:2], self.Bm[:2], self.Cm[:2]):
                self.pontuacao += 10
                continue

            if self.ponto_no_triangulo(intersecao[:2], self.A[:2], self.B[:2], self.C[:2]):
                self.pontuacao += 5

        print(f"\nPontuação final: {self.pontuacao}")

    def visualizar(self):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        triangulo_maior = np.array([self.A, self.B, self.C, self.A])
        ax.plot(triangulo_maior[:, 0], triangulo_maior[:, 1], triangulo_maior[:, 2], 'b-', label='Triângulo maior (5 pts)')

        triangulo_menor = np.array([self.Am, self.Bm, self.Cm, self.Am])
        ax.plot(triangulo_menor[:, 0], triangulo_menor[:, 1], triangulo_menor[:, 2], 'g-', label='Triângulo menor (10 pts)')

        ax.scatter(*self.centro, color='red', s=100, label='Centro (20 pts)')
        ax.scatter(*self.lancador, color='black', s=100, label='Lançador')

        for i, (direcao, intersecao) in enumerate(zip(self.vetores_direcao, self.pontos_interseccao)):
            fim = self.lancador + direcao * 2

            if intersecao is not None:
                ax.plot([self.lancador[0], fim[0]], [self.lancador[1], fim[1]], [self.lancador[2], fim[2]], 'k--', alpha=0.3)
                ax.scatter(*intersecao, color='orange', s=50)
            else:
                ax.plot([self.lancador[0], fim[0]], [self.lancador[1], fim[1]], [self.lancador[2], fim[2]], 'r--', alpha=0.5, label='Erro' if i == 0 else "")

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Lançamento de Dardos - Visualização 3D')
        ax.legend()
        plt.tight_layout()
        plt.show()

def principal():
    jogo = JogoDeDardos()

    # Exemplo de entrada manual (descomente para usar)
    # jogo.ler_entrada()

    # Exemplo direto
    jogo.A = np.array([0, 7, 2])
    jogo.B = np.array([11, 0, 2])
    jogo.C = np.array([5.35239, 3.57801, 6.68146])
    jogo.lancador = np.array([0, 0, 4])
    jogo.num_lancamentos = 1
    jogo.vetores_direcao = [np.array([1, 1, 4])]

    try:
        jogo.validar_entrada()
        jogo.jogar()
        jogo.visualizar()
    except ValueError as e:
        print(f"Erro: {e}")

if __name__ == "__main__":
    principal()
