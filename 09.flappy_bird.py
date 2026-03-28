# ==========================================================
# [ FLAPPY BIRD GAME MANUAL ]
# ==========================================================
# 1. 파일 세팅: 파이썬 파일과 같은 위치에 '09.img' 폴더 생성
# 2. 필수 리소스 (09.img 폴더 안에 넣기):
#    - 이미지: bird.png, background.png, pipe.png
#    - 사운드: move.mp3, game_over.mp3, score.mp3
# 3. 조작 방법: [SPACE] 키 하나로 시작, 점프, 재시작 가능
# ==========================================================

import pygame
import random
import os

# 1. 초기화
pygame.init()
pygame.mixer.init()

# 화면 설정
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.HWACCEL)
pygame.display.set_caption("Ultimate Flappy Bird")
clock = pygame.time.Clock()

# 리소스 폴더 설정
ASSET_PATH = "09.img"

# 색상 및 폰트 설정
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 50, 50)
YELLOW = (255, 215, 0)

font_score = pygame.font.SysFont(None, 70, bold=True)
font_big = pygame.font.SysFont(None, 45, bold=True)
font_msg = pygame.font.SysFont(None, 35, bold=True)

# 2. 리소스 로드 함수
def get_path(filename):
    return os.path.join(ASSET_PATH, filename)

def load_image(file, width, height):
    path = get_path(file)
    if os.path.exists(path):
        img = pygame.image.load(path).convert_alpha()
        return pygame.transform.scale(img, (width, height))
    surf = pygame.Surface((width, height))
    surf.fill((255, 215, 0))
    return surf

def load_sound(file):
    path = get_path(file)
    if os.path.exists(path):
        return pygame.mixer.Sound(path)
    return None

# 리소스 불러오기
background_img = load_image('background.png', SCREEN_WIDTH, SCREEN_HEIGHT)
bird_img = load_image('bird.png', 35, 25)

PIPE_WIDTH = 80
PIPE_HEIGHT = SCREEN_HEIGHT
pipe_base_img = load_image('pipe.png', PIPE_WIDTH, PIPE_HEIGHT)
pipe_bottom_img = pipe_base_img
pipe_top_img = pygame.transform.flip(pipe_base_img, False, True)

score_sound = load_sound('score.mp3') or load_sound('flappy_bird_score.mp3')
move_sound = load_sound('move.mp3')
die_sound = load_sound('game_over.mp3')

# 3. 게임 시스템 함수
def reset_game():
    global bird_pos, bird_vel, pipes, last_pipe, game_over, score, bg_scroll
    bird_pos = [100, 300]
    bird_vel = 0
    pipes = []
    last_pipe = pygame.time.get_ticks()
    game_over = False
    score = 0
    bg_scroll = 0

def fade_out():
    fade_surf = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    fade_surf.fill(BLACK)
    for alpha in range(0, 256, 15):
        fade_surf.set_alpha(alpha)
        screen.blit(background_img, (0, 0))
        screen.blit(fade_surf, (0, 0))
        pygame.display.flip()
        clock.tick(60)

def draw_text_with_shadow(text, font, text_color, shadow_color, center_pos):
    shadow_surf = font.render(text, True, shadow_color)
    shadow_rect = shadow_surf.get_rect(center=(center_pos[0] + 3, center_pos[1] + 3))
    screen.blit(shadow_surf, shadow_rect)
    text_surf = font.render(text, True, text_color)
    text_rect = text_surf.get_rect(center=center_pos)
    screen.blit(text_surf, text_rect)

# 4. 초기 상태 설정
reset_game()
gravity = 0.25
jump_strength = -6
pipe_frequency = 1600
game_state = 'READY'
bg_scroll = 0
bg_scroll_speed = 1

# 5. 메인 루프
running = True
while running:
    # 배경 스크롤
    if not game_over:
        bg_scroll -= bg_scroll_speed
        if bg_scroll <= -SCREEN_WIDTH:
            bg_scroll = 0
            
    screen.blit(background_img, (bg_scroll, 0))
    screen.blit(background_img, (bg_scroll + SCREEN_WIDTH, 0))

    # 이벤트 처리
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            if game_state == 'READY':
                reset_game()
                game_state = 'PLAYING'
                bird_vel = jump_strength
                if move_sound: move_sound.play()
            elif game_state == 'PLAYING' and not game_over:
                bird_vel = jump_strength
                if move_sound:
                    move_sound.stop()
                    move_sound.play()
            elif game_over:
                fade_out()
                reset_game()
                game_state = 'READY'

    bird_rect = bird_img.get_rect(center=(bird_pos[0], bird_pos[1]))

    if game_state == 'READY':
        draw_text_with_shadow("PRESS SPACE TO START", font_msg, WHITE, BLACK, (SCREEN_WIDTH//2, SCREEN_HEIGHT*0.8))

    elif game_state == 'PLAYING':
        if not game_over:
            bird_vel += gravity
            bird_pos[1] += bird_vel

            current_time = pygame.time.get_ticks()
            if current_time - last_pipe > pipe_frequency:
                top_h = random.randint(100, 350)
                gap = 150
                top_rect = pygame.Rect(SCREEN_WIDTH, top_h - PIPE_HEIGHT, PIPE_WIDTH, PIPE_HEIGHT)
                bot_rect = pygame.Rect(SCREEN_WIDTH, top_h + gap, PIPE_WIDTH, PIPE_HEIGHT)
                pipes.append([top_rect, bot_rect, False])
                last_pipe = current_time

            for p in pipes:
                p[0].x -= 3
                p[1].x -= 3
                
                # 히트박스 보정 (억울한 죽음 방지)
                bird_hitbox = bird_rect.inflate(-12, -12)
                top_pipe_hitbox = p[0].inflate(-15, 0)
                bot_pipe_hitbox = p[1].inflate(-15, 0)

                if not p[2] and bird_pos[0] > p[0].right:
                    score += 1
                    p[2] = True 
                    if score_sound: score_sound.play()

                screen.blit(pipe_top_img, p[0])
                screen.blit(pipe_bottom_img, p[1])
                
                if bird_hitbox.colliderect(top_pipe_hitbox) or bird_hitbox.colliderect(bot_pipe_hitbox):
                    if not game_over and die_sound: die_sound.play()
                    game_over = True

            pipes = [p for p in pipes if p[0].right > 0]

            if bird_pos[1] <= 0 or bird_pos[1] >= SCREEN_HEIGHT:
                if not game_over and die_sound: die_sound.play()
                game_over = True

        draw_text_with_shadow(str(score), font_score, WHITE, BLACK, (SCREEN_WIDTH // 2, 70))

        if game_over:
            over_surf = font_big.render("GAME OVER!", True, RED)
            over_rect = over_surf.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 - 40))
            screen.blit(over_surf, over_rect)
            draw_text_with_shadow("PRESS SPACE TO RESTART", font_msg, YELLOW, BLACK, (SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 30))

    # 새 그리기
    if game_state == 'READY':
        screen.blit(bird_img, bird_img.get_rect(center=(bird_pos[0], bird_pos[1])))
    else:
        rot_bird = pygame.transform.rotozoom(bird_img, -bird_vel * 3, 1)
        screen.blit(rot_bird, rot_bird.get_rect(center=bird_rect.center))
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()