import pygame
import sys
import random

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

def game_loop():
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

    while not game_over:
        current_time = pygame.time.get_ticks()

        if current_time - last_obstacle_time > obstacle_interval:
            last_obstacle_time = current_time
            obstacle_height = random.randint(20, 70)
            obstacles.append(Obstacle(width, height - obstacle_height - ground_height, obstacle_width, obstacle_height))

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
            if dino.rect.colliderect(obstacle.rect):
                game_over = True
                break
            if obstacle.x < -obstacle_width:
                obstacles.remove(obstacle)
                score += 1

        screen.fill(white)
        dino.draw(screen, green)
        for obstacle in obstacles:
            obstacle.draw(screen, red)
        draw_text(f'Score: {score}', pygame.font.Font(None, 36), black, screen, 10, 10)
        pygame.display.flip()
        clock.tick(60)

    while game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    game_loop()

        screen.fill(white)
        draw_text('Game Over', pygame.font.Font(None, 48), red, screen, width // 2 - 100, height // 2 - 20)
        draw_text('Press Space to Restart', pygame.font.Font(None, 36), black, screen, width // 2 - 140, height // 2 + 30)
        pygame.display.flip()
        clock.tick(60)

# Initialize Pygame
pygame.init()
game_loop()
