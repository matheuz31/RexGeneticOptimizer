import pygame
import torch
import sys
import random
from neural_network import DQN, ReplayMemory
from training import select_action, optimize_model
import time
import torch.optim as optim



class Dinosaur:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.vel_y = 0
        self.jump = False
        self.rect = pygame.Rect(x, y, width, height)

    def update(self, gravity, ground_height, jump_height, height):
        if self.jump:
            self.y += self.vel_y
            self.vel_y += gravity
            if self.y > height - self.height - ground_height:
                self.y = height - self.height - ground_height
                self.jump = False
        self.rect.y = self.y

    def draw(self, screen, color):
        pygame.draw.rect(screen, color, self.rect)

class Obstacle:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.rect = pygame.Rect(x, y, width, height)

    def update(self, game_speed):
        self.x -= game_speed
        self.rect.x = self.x

    def draw(self, screen, color):
        pygame.draw.rect(screen, color, self.rect)

def draw_text(text, font, color, surface, x, y):
    textobj = font.render(text, True, color)
    textrect = textobj.get_rect()
    textrect.topleft = (x, y)
    surface.blit(textobj, textrect)

def game_loop(policy_net, target_net, optimizer, memory, device):
    # Game settings
    width, height = 800, 400
    white, black, green, red = (255, 255, 255), (0, 0, 0), (0, 255, 0), (255, 0, 0)
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Dinosaur Game")
    clock = pygame.time.Clock()
    gravity, ground_height, jump_height = 1, 20, -15
    obstacle_interval, obstacle_width = 1500, 20
    last_obstacle_time, score, game_speed = pygame.time.get_ticks(), 0, 5
    dino = Dinosaur(50, height - 60, 20, 40)
    obstacles = []
    game_over = False

    n_actions = 2

    while not game_over:
        current_time = pygame.time.get_ticks()

        if current_time - last_obstacle_time > obstacle_interval:
            last_obstacle_time = current_time
            obstacle_height = random.randint(20, 70)
            obstacles.append(Obstacle(width, height - obstacle_height - ground_height, obstacle_width, obstacle_height))

        state = get_state(dino, obstacles, game_speed, height, width)
        action = select_action(state, n_actions, policy_net, device)

        if action.item() == 1 and not dino.jump:
            dino.jump = True
            dino.vel_y = jump_height


        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and not dino.jump:
                    dino.jump = True
                    dino.vel_y = jump_height

        
        dino.update(gravity, ground_height, jump_height, height)
        for obstacle in obstacles:
            obstacle.update(game_speed)

        if not game_over:
            next_state = get_state(dino, obstacles, game_speed, height, width)

        reward = 0  # Recompensa padrão por frame

        for obstacle in obstacles:
            obstacle.update(game_speed)
            if dino.rect.colliderect(obstacle.rect):
                reward = -10  # Penalidade por colidir
                game_over = True
                break
            elif obstacle.x < -obstacle_width:
                obstacles.remove(obstacle)
                score += 1
                reward = 1  # Recompensa por passar um obstáculo

        memory.push(state, action, next_state, torch.tensor([reward], device=device))
        optimize_model(policy_net, target_net, optimizer, memory, device)
    

        screen.fill(white)
        dino.draw(screen, green)
        for obstacle in obstacles:
            obstacle.draw(screen, red)
        draw_text(f'Score: {score}', pygame.font.Font(None, 36), black, screen, 10, 10)
        pygame.display.flip()
        clock.tick(60)
    
    print(f"Episódio terminou com pontuação: {score}")

    time.sleep(1)  # Atraso antes de reiniciar o jogo, por exemplo, 1 segundo
    return
    

    # while game_over:
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             pygame.quit()
    #             sys.exit()
    #         if event.type == pygame.KEYDOWN:
    #             if event.key == pygame.K_SPACE:
    #                 game_loop()

    #     screen.fill(white)
    #     draw_text('Game Over', pygame.font.Font(None, 48), red, screen, width // 2 - 100, height // 2 - 20)
    #     draw_text('Press Space to Restart', pygame.font.Font(None, 36), black, screen, width // 2 - 140, height // 2 + 30)
    #     pygame.display.flip()
    #     clock.tick(60)

def get_state(dino, obstacles, game_speed, width, height):
    if obstacles:
        next_obstacle = obstacles[0]
        distance_to_next_obstacle = next_obstacle.x - dino.x
        height_of_next_obstacle = next_obstacle.height
    else:
        distance_to_next_obstacle = width  # Uma distância padrão grande se não houver obstáculos
        height_of_next_obstacle = 0

    # Normalização (ajuste estes valores conforme necessário)
    normalized_distance = distance_to_next_obstacle / width
    normalized_height = height_of_next_obstacle / height
    normalized_game_speed = game_speed / 10  # Supondo que 10 seja a velocidade máxima

    # A posição vertical do dinossauro pode ser importante se você quiser que o modelo aprenda quando não pular
    normalized_dino_y = dino.y / height

    state = [normalized_distance, normalized_height, normalized_game_speed, normalized_dino_y]
    return torch.tensor([state], dtype=torch.float)


if __name__ == "__main__":

    pygame.init()

    input_size = 4  # Número de características do estado
    hidden_size = 256
    output_size = 2  # Número de ações possíveis
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_net = DQN(input_size, hidden_size, output_size).to(device)
    target_net = DQN(input_size, hidden_size, output_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()  # A target_net está sempre em modo de avaliação

    optimizer = optim.RMSprop(policy_net.parameters())
    memory = ReplayMemory(10000)

    while True:  # Loop para reiniciar o jogo
        game_loop(policy_net, target_net, optimizer, memory, device)
    