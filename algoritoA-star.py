import pygame
import math
from queue import PriorityQueue

# Configuración de ventana
ANCHO = 800
FILAS = 11
VENTANA = pygame.display.set_mode((ANCHO, ANCHO))
pygame.display.set_caption("A* Pathfinding con Pesos")

# Colores
BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)
GRIS = (128, 128, 128)
VERDE = (0, 255, 0)
ROJO = (255, 0, 0)
NARANJA = (255, 165, 0)
PURPURA = (128, 0, 128)
TURQUESA = (64, 224, 208)

# Clase Nodo
class Nodo:
    def __init__(self, fila, col, ancho, total_filas):
        self.fila = fila
        self.col = col
        self.x = col * ancho
        self.y = fila * ancho
        self.color = BLANCO
        self.ancho = ancho
        self.total_filas = total_filas
        self.vecinos = []

    def get_pos(self):
        return self.fila, self.col

    def es_pared(self):
        return self.color == NEGRO

    def es_inicio(self):
        return self.color == NARANJA

    def es_fin(self):
        return self.color == PURPURA

    def restablecer(self):
        self.color = BLANCO

    def hacer_inicio(self):
        self.color = NARANJA

    def hacer_pared(self):
        self.color = NEGRO

    def hacer_fin(self):
        self.color = PURPURA

    def hacer_abierto(self):
        self.color = VERDE

    def hacer_cerrado(self):
        self.color = ROJO

    def hacer_camino(self):
        self.color = TURQUESA

    def dibujar(self, ventana):
        pygame.draw.rect(ventana, self.color, (self.x, self.y, self.ancho, self.ancho))

    def actualizar_vecinos(self, grid):
        self.vecinos = []
        # Direcciones con pesos (adyacentes = 10, diagonales = 14)
        direcciones = [
            (-1, 0, 10), (1, 0, 10), (0, -1, 10), (0, 1, 10),  # rectos
            (-1, -1, 14), (-1, 1, 14), (1, -1, 14), (1, 1, 14)  # diagonales
        ]
        for dx, dy, peso in direcciones:
            fila = self.fila + dx
            col = self.col + dy
            if 0 <= fila < self.total_filas and 0 <= col < self.total_filas:
                vecino = grid[fila][col]
                if not vecino.es_pared():
                    self.vecinos.append((vecino, peso))

    def __lt__(self, otro):
        return False

# Heurística (distancia Manhattan)
def h(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)

# Reconstrucción del camino
def reconstruir_camino(came_from, actual, dibujar):
    while actual in came_from:
        actual = came_from[actual]
        actual.hacer_camino()
        dibujar()

# Algoritmo A*
def a_estrella(dibujar, grid, inicio, fin):
    contador = 0
    abierta = PriorityQueue()
    abierta.put((0, contador, inicio))
    came_from = {}

    g_score = {nodo: float("inf") for fila in grid for nodo in fila}
    g_score[inicio] = 0

    f_score = {nodo: float("inf") for fila in grid for nodo in fila}
    f_score[inicio] = h(inicio.get_pos(), fin.get_pos())

    abierta_hash = {inicio}

    while not abierta.empty():
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                pygame.quit()

        actual = abierta.get()[2]
        abierta_hash.remove(actual)

        if actual == fin:
            reconstruir_camino(came_from, fin, dibujar)
            fin.hacer_fin()
            inicio.hacer_inicio()
            return True

        for vecino, peso in actual.vecinos:
            temp_g_score = g_score[actual] + peso

            if temp_g_score < g_score[vecino]:
                came_from[vecino] = actual
                g_score[vecino] = temp_g_score
                f_score[vecino] = temp_g_score + h(vecino.get_pos(), fin.get_pos())
                if vecino not in abierta_hash:
                    contador += 1
                    abierta.put((f_score[vecino], contador, vecino))
                    abierta_hash.add(vecino)
                    vecino.hacer_abierto()

        dibujar()
        if actual != inicio:
            actual.hacer_cerrado()

    return False

# Crear grid
def crear_grid(filas, ancho):
    grid = []
    ancho_nodo = ancho // filas
    for i in range(filas):
        grid.append([])
        for j in range(filas):
            nodo = Nodo(i, j, ancho_nodo, filas)
            grid[i].append(nodo)
    return grid

# Dibujar líneas
def dibujar_grid(ventana, filas, ancho):
    ancho_nodo = ancho // filas
    for i in range(filas):
        pygame.draw.line(ventana, GRIS, (0, i * ancho_nodo), (ancho, i * ancho_nodo))
        for j in range(filas):
            pygame.draw.line(ventana, GRIS, (j * ancho_nodo, 0), (j * ancho_nodo, ancho))

# Dibujar todo
def dibujar(ventana, grid, filas, ancho):
    ventana.fill(BLANCO)
    for fila in grid:
        for nodo in fila:
            nodo.dibujar(ventana)

    dibujar_grid(ventana, filas, ancho)
    pygame.display.update()

# Obtener posición del clic
def obtener_click_pos(pos, filas, ancho):
    ancho_nodo = ancho // filas
    x, y = pos
    fila = y // ancho_nodo
    col = x // ancho_nodo
    return fila, col

# Función principal
def main(ventana, ancho):
    grid = crear_grid(FILAS, ancho)
    inicio = None
    fin = None
    corriendo = True

    while corriendo:
        dibujar(ventana, grid, FILAS, ancho)
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                corriendo = False

            if pygame.mouse.get_pressed()[0]:  # Clic izquierdo
                pos = pygame.mouse.get_pos()
                fila, col = obtener_click_pos(pos, FILAS, ancho)
                nodo = grid[fila][col]
                if not inicio and nodo != fin:
                    inicio = nodo
                    inicio.hacer_inicio()
                elif not fin and nodo != inicio:
                    fin = nodo
                    fin.hacer_fin()
                elif nodo != fin and nodo != inicio:
                    nodo.hacer_pared()

            elif pygame.mouse.get_pressed()[2]:  # Clic derecho
                pos = pygame.mouse.get_pos()
                fila, col = obtener_click_pos(pos, FILAS, ancho)
                nodo = grid[fila][col]
                nodo.restablecer()
                if nodo == inicio:
                    inicio = None
                elif nodo == fin:
                    fin = None

            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_SPACE and inicio and fin:
                    for fila in grid:
                        for nodo in fila:
                            nodo.actualizar_vecinos(grid)
                    a_estrella(lambda: dibujar(ventana, grid, FILAS, ancho), grid, inicio, fin)

                if evento.key == pygame.K_c:
                    inicio = None
                    fin = None
                    grid = crear_grid(FILAS, ancho)

    pygame.quit()

# Ejecutar
main(VENTANA, ANCHO)
