import pygame
import sys
import random
import math

pygame.init()

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 500
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Balls Simulation")

# Simulation setup
sim_min_width = 20.0
c_scale = min(WINDOW_WIDTH, WINDOW_HEIGHT) / sim_min_width
sim_width = WINDOW_WIDTH / c_scale
sim_height = WINDOW_HEIGHT / c_scale

def c_x(pos_x):
    return int(pos_x * c_scale)

def c_y(pos_y):
    return int(WINDOW_HEIGHT - pos_y * c_scale)

def screen_to_sim_x(screen_x):
    return screen_x / c_scale

def screen_to_sim_y(screen_y):
    return (WINDOW_HEIGHT - screen_y) / c_scale

# Scene setup
gravity = {'x': 0.0, 'y': -10.0}
time_step = 1.0 / 60.0
air_resistance = 0.02 
restitution = 0.9 


# Create multiple balls
def create_ball(x, y, vx, vy, color):
    """Create a ball with given properties"""
    return {
        'radius': 1.0,
        'pos': {'x': x, 'y': y},
        'vel': {'x': vx, 'y': vy},
        'color': color
    }

balls = []
for i in range(10):
    x = random.uniform(1.0, sim_width - 1.0)
    y = random.uniform(2.0, sim_height - 2.0)
    vx = random.uniform(-8.0, 8.0)
    vy = random.uniform(-5.0, 15.0)
    color = (255, 0, 0)
    balls.append(create_ball(x, y, vx, vy, color))



# Colors
RED = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Clock for controlling frame rate
clock = pygame.time.Clock()


# Draw a scene
def draw():
    screen.fill(WHITE)
    
    for ball in balls:
        pygame.draw.circle(
            screen,
            ball['color'],
            (c_x(ball['pos']['x']), c_y(ball['pos']['y'])),
            int(c_scale * ball['radius'])
        )  
    pygame.display.flip()

def apply_air_resistance(ball):
    speed = math.sqrt(ball['vel']['x']**2 + ball['vel']['y']**2)
    if speed > 0:
        ball['vel']['x'] -= ball['vel']['x'] * speed * time_step * air_resistance
        ball['vel']['y'] -= ball['vel']['y'] * speed * time_step * air_resistance

# ball colision, calculate distance and handle collision
def handle_ball_collision(ball1, ball2):
    # Calculate distance between ball centers
    dx = ball2['pos']['x'] - ball1['pos']['x']
    dy = ball2['pos']['y'] - ball1['pos']['y']
    distance = math.sqrt(dx**2 + dy**2)
    
    # Check if balls are colliding
    min_distance = ball1['radius'] + ball2['radius']
    if distance < min_distance and distance > 0.001:  # Avoid division by zero
        # Normalize collision vector
        nx = dx / distance
        ny = dy / distance
        # Separate balls to prevent overlap
        overlap = min_distance - distance
        ball1['pos']['x'] -= nx * overlap * 0.5
        ball1['pos']['y'] -= ny * overlap * 0.5
        ball2['pos']['x'] += nx * overlap * 0.5
        ball2['pos']['y'] += ny * overlap * 0.5
        
        rel_vel_x = ball1['vel']['x'] - ball2['vel']['x']
        rel_vel_y = ball1['vel']['y'] - ball2['vel']['y']
        vel_along_normal = rel_vel_x * nx + rel_vel_y * ny
        
        # Only if balls are moving towards each other
        if vel_along_normal > 0:
            impulse = vel_along_normal * (1.0 + restitution) * 0.5
            ball1['vel']['x'] -= impulse * nx
            ball1['vel']['y'] -= impulse * ny
            ball2['vel']['x'] += impulse * nx
            ball2['vel']['y'] += impulse * ny

def simulate():
    for ball in balls:
        # Apply gravity
        ball['vel']['x'] += gravity['x'] * time_step
        ball['vel']['y'] += gravity['y'] * time_step
        
        # Apply air resistance
        apply_air_resistance(ball)
        
        # Update position
        ball['pos']['x'] += ball['vel']['x'] * time_step
        ball['pos']['y'] += ball['vel']['y'] * time_step
        
        # Boundary collisions
        if ball['pos']['x'] - ball['radius'] < 0.0:
            ball['pos']['x'] = ball['radius']
            ball['vel']['x'] = -ball['vel']['x'] * restitution
        
        if ball['pos']['x'] + ball['radius'] > sim_width:
            ball['pos']['x'] = sim_width - ball['radius']
            ball['vel']['x'] = -ball['vel']['x'] * restitution
        
        if ball['pos']['y'] - ball['radius'] < 0.0:
            ball['pos']['y'] = ball['radius']
            ball['vel']['y'] = -ball['vel']['y'] * restitution
        
        if ball['pos']['y'] + ball['radius'] > sim_height:
            ball['pos']['y'] = sim_height - ball['radius']
            ball['vel']['y'] = -ball['vel']['y'] * restitution

    
    # Handle ball-ball collisions
    for i in range(len(balls)):
        for j in range(i + 1, len(balls)):
            handle_ball_collision(balls[i], balls[j])

#mouse click to hit a ball
def handle_mouse_click(mouse_x, mouse_y):
    sim_x = screen_to_sim_x(mouse_x)
    sim_y = screen_to_sim_y(mouse_y)
    
    # Check if click is on any ball
    for ball in balls:
        dx = ball['pos']['x'] - sim_x
        dy = ball['pos']['y'] - sim_y
        distance = math.sqrt(dx**2 + dy**2)
        
        if distance <= ball['radius']:
            ball['vel']['y'] += 15.0  # Add velocity upward
            break  

# Main game loop
def main():
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    handle_mouse_click(mouse_x, mouse_y)
        simulate()        
        draw()
        clock.tick(60)
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
