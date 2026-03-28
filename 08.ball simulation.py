'''
[사용 방법]
1. 화면이 뜨면 키보드로 원하는 공의 최대 개수(최대 15,000)를 입력하고 엔터를 누르세요.
2. 시뮬레이션이 시작되며, 목표 개수에 도달 후 5초 뒤 자동으로 첫 화면으로 리셋됩니다.
'''

import pygame
import math
import random
import numpy as np
import sys

WIDTH, HEIGHT = 800, 800
CIRCLE_RADIUS = 350
CIRCLE_CENTER = (WIDTH // 2, HEIGHT // 2)
BALL_RADIUS = 8
FPS = 60

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 230, 0) 

TRAIL_ALPHA = 70 
SPEED_MIN = 4
SPEED_MAX = 7

def create_bounce_sound():
    sample_rate = 44100
    duration = 0.1
    frequency = 440.0
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = np.sin(2 * np.pi * frequency * t)
    
    fade_out = np.linspace(1, 0, len(wave))
    wave = (wave * fade_out * 32767).astype(np.int16)
    
    stereo_wave = np.vstack((wave, wave)).T.copy(order='C')
    return pygame.sndarray.make_sound(stereo_wave)

class Ball:
    def __init__(self, x, y, dx, dy, color):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.color = color
        self.can_spawn = True 

    def move(self):
        self.x += self.dx
        self.y += self.dy

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), BALL_RADIUS)

    def check_collision(self):
        dist = math.sqrt((self.x - CIRCLE_CENTER[0])**2 + (self.y - CIRCLE_CENTER[1])**2)
        
        if dist + BALL_RADIUS >= CIRCLE_RADIUS:
            nx = (self.x - CIRCLE_CENTER[0]) / dist
            ny = (self.y - CIRCLE_CENTER[1]) / dist
            
            dot_product = self.dx * nx + self.dy * ny
            
            self.dx -= 2 * dot_product * nx
            self.dy -= 2 * dot_product * ny
            
            overlap = (dist + BALL_RADIUS) - CIRCLE_RADIUS
            self.x -= nx * (overlap + 1)
            self.y -= ny * (overlap + 1)

            if self.can_spawn:
                self.can_spawn = False
                return True
        else:
            if dist + BALL_RADIUS < CIRCLE_RADIUS - 15:
                self.can_spawn = True
        return False

def get_korean_font(size):
    font_names = ['applegothic', 'malgungothic', 'nanumgothic', 'dotum', 'gulim']
    for font_name in font_names:
        if pygame.font.match_font(font_name):
            return pygame.font.SysFont(font_name, size)
    return pygame.font.SysFont(None, size)

def get_polarized_speed():
    speed = random.uniform(SPEED_MIN, SPEED_MAX)
    direction = random.choice([-1, 1])
    return speed * direction

def get_max_ball_count(screen, clock, font, trail_overlay):
    input_text = ""
    prompt1 = font.render("원하는 공의 개수를 입력하세요 (최대 15,000)", True, WHITE)
    
    cursor_visible = True
    last_blink_time = pygame.time.get_ticks()

    while True:
        screen.fill(BLACK)
        screen.blit(prompt1, (WIDTH // 2 - prompt1.get_width() // 2, HEIGHT // 2 - 60))
        
        current_time = pygame.time.get_ticks()
        if current_time - last_blink_time > 500:
            cursor_visible = not cursor_visible
            last_blink_time = current_time
        
        display_text = input_text + ("|" if cursor_visible else "")
        text_surface = font.render(display_text, True, YELLOW)
        
        screen.blit(text_surface, (WIDTH // 2 - text_surface.get_width() // 2, HEIGHT // 2))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    if input_text.isdigit():
                        num = int(input_text)
                        if num <= 0:
                            num = 1
                        return num
                elif event.key == pygame.K_BACKSPACE:
                    input_text = input_text[:-1]
                else:
                    if event.unicode.isdigit():
                        temp_text = input_text + event.unicode
                        if int(temp_text) > 15000:
                            input_text = "15000"
                        else:
                            input_text = temp_text
        
        clock.tick(FPS)

def main():
    pygame.init()
    pygame.mixer.init() 
    
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("구슬 증식 시뮬레이션")
    clock = pygame.time.Clock()
    
    font = get_korean_font(32)
    small_font = get_korean_font(24)

    bounce_sound = create_bounce_sound()
    bounce_sound.set_volume(0.1) 

    trail_overlay = pygame.Surface((WIDTH, HEIGHT))
    trail_overlay.set_alpha(TRAIL_ALPHA) 
    trail_overlay.fill(BLACK)

    while True:
        max_balls = get_max_ball_count(screen, clock, font, trail_overlay)

        dx = get_polarized_speed()
        dy = get_polarized_speed()
        color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
        balls = [Ball(WIDTH//2, HEIGHT//2, dx, dy, color)]

        timer_active = False
        start_ticks = 0
        simulation_active = True 
        waiting_for_restart = False
        stop_ticks = 0
        
        running = True
        while running:
            screen.blit(trail_overlay, (0, 0))
            pygame.draw.circle(screen, WHITE, CIRCLE_CENTER, CIRCLE_RADIUS, 2)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            if simulation_active:
                new_balls_to_add = []
                sounds_played_this_frame = 0 
                
                for ball in balls:
                    ball.move() 
                    if ball.check_collision(): 
                        if sounds_played_this_frame < 3:
                            bounce_sound.play()
                            sounds_played_this_frame += 1
                        
                        new_dx = get_polarized_speed()
                        new_dy = get_polarized_speed()
                        new_color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
                        new_ball = Ball(ball.x, ball.y, new_dx, new_dy, new_color)
                        new_ball.can_spawn = False 
                        new_balls_to_add.append(new_ball)
                        
                balls.extend(new_balls_to_add)

                if len(balls) > max_balls:
                    balls = balls[:max_balls]

                if len(balls) >= max_balls and not timer_active:
                    timer_active = True
                    start_ticks = pygame.time.get_ticks() 

                if timer_active:
                    elapsed_time = (pygame.time.get_ticks() - start_ticks) / 1000.0
                    if elapsed_time >= 5.0:
                        simulation_active = False
                        waiting_for_restart = True
                        stop_ticks = pygame.time.get_ticks() 
            else:
                if waiting_for_restart:
                    elapsed_stop_time = (pygame.time.get_ticks() - stop_ticks) / 1000.0
                    if elapsed_stop_time >= 1.0:
                        running = False 

            for ball in balls:
                ball.draw(screen)

            count_text = small_font.render(f"현재 공 개수: {len(balls):,}", True, WHITE)
            screen.blit(count_text, (20, 20))

            target_shadow = small_font.render(f"목표치: {max_balls:,}", True, (100, 80, 0)) 
            target_text = small_font.render(f"목표치: {max_balls:,}", True, YELLOW)
            screen.blit(target_shadow, (22, 52)) 
            screen.blit(target_text, (20, 50))   

            if not simulation_active:
                stop_text = small_font.render("재시작 준비 중...", True, (255, 100, 100))
                screen.blit(stop_text, (20, 90))

            pygame.display.flip() 
            clock.tick(FPS) 

if __name__ == "__main__":
    main()