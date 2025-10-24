from __future__ import annotations
import pygame
import sys
import random
import math
from dataclasses import dataclass

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


@dataclass
class Vector2:
    x: float = 0.0
    y: float = 0.0
    def set(self, v: Vector2):
        self.x = v.x
        self.y = v.y
    
    def clone(self):
        return Vector2(self.x, self.y)
    
    def add(self, v: Vector2, scale: float = 1.0):
        self.x=self.x + v.x * scale
        self.y=self.y + v.y * scale
        return self
    
    def addVectors(self, a:Vector2, b:Vector2):
        self.x = a.x + b.x 
        self.y = a.y + b.y
        return self
    
    def substract(self, v: Vector2, scale: float = 1.0):
        self.x=self.x - v.x * scale
        self.y=self.y - v.y * scale
        return self
    def substractVectors(self, a:Vector2, b:Vector2):
        self.x = a.x - b.x 
        self.y = a.y - b.y
        return self

    def length(self):
        return math.sqrt(self.x * self.x + self.y * self.y)
        
    def scale(self, s: float):
        self.x=self.x * s
        self.y=self.y * s
        return self
    def dot(self, v: Vector2):
        return self.x * v.x + self.y * v.y
    
    def perp(self):
        return Vector2(-self.y, self.x)

@dataclass
class Ball:
    radius: float
    mass : float
    pos: Vector2
    vel: Vector2
    def simulate(self, dt: float, gravity: Vector2):
        self.vel.add(gravity, dt)
        self.pos.add(self.vel, dt)

# Scene setup

@dataclass
class PhysicsScene:
    gravity: Vector2
    dt: float
    worldSize:Vector2
    balls: list[Ball] = None
    restitution: float = 1.0
    
physicScene = PhysicsScene(
    gravity=Vector2(0.0, -10.0),
    dt=1.0 / 60.0,
    worldSize=Vector2(sim_width, sim_height),
    balls=[],
    restitution=0.9
)




def setupScene():

    physicScene.balls.clear()
    num_balls = 10
    
    for _ in range(num_balls):
        radius = random.uniform(0.3, 0.7)
        mass = math.pi * radius * radius
        pos = Vector2(
            random.uniform(radius, physicScene.worldSize.x - radius),
            random.uniform(radius, physicScene.worldSize.y - radius)
        )
        vel = Vector2(
            random.uniform(-5.0, 5.0),
            random.uniform(-5.0, 5.0)
        )
        physicScene.balls.append(Ball(radius=radius, mass=mass, pos=pos, vel=vel))

        
    
setupScene()


# Colors
RED = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Clock for controlling frame rate
clock = pygame.time.Clock()


# Draw a scene
def draw():
    screen.fill(WHITE)
    
    for ball in physicScene.balls:
        pygame.draw.circle(
            screen,
            RED,
            (c_x(ball.pos.x), c_y(ball.pos.y)),
            int(c_scale * ball.radius)
        )  
    pygame.display.flip()
    
    

# ball colision, calculate distance and handle collision
def handle_ball_collision(ball1, ball2):
    # Calculate distance between ball centers
    
    diri = Vector2()
    diri.substractVectors(ball2.pos, ball1.pos)
    
    d= diri.length()
    if (d==0 or d > ball1.radius + ball2.radius):
        return
    
    diri.scale(1.0/d)
    
    corr = (ball1.radius + ball2.radius - d) / 2.0
    
    ball1.pos.add(diri, -corr)
    ball2.pos.add(diri, corr)
    
    v1 = ball1.vel.dot(diri)
    v2 = ball2.vel.dot(diri)
    
    m1 = ball1.mass
    m2 = ball2.mass
    
    newv1 = (m1 * v1 + m2 * v2 - m2 * (v1 - v2) * physicScene.restitution) / (m1 + m2)
    newv2 = (m1 * v1 + m2 * v2 - m1 * (v2 - v1) * physicScene.restitution) / (m1 + m2)
    ball1.vel.add(diri, newv1 - v1)
    ball2.vel.add(diri, newv2 - v2)
    
def handle_wall_collision(ball, worldSize):
    if ball.pos.x - ball.radius < 0.0:
        ball.pos.x = ball.radius
        ball.vel.x = -ball.vel.x * physicScene.restitution
    
    if ball.pos.x + ball.radius > worldSize.x:
        ball.pos.x = worldSize.x - ball.radius
        ball.vel.x = -ball.vel.x * physicScene.restitution

    if ball.pos.y - ball.radius < 0.0:
        ball.pos.y = ball.radius
        ball.vel.y = -ball.vel.y * physicScene.restitution

    if ball.pos.y + ball.radius > worldSize.y:
        ball.pos.y = worldSize.y - ball.radius
        ball.vel.y = -ball.vel.y * physicScene.restitution
    
def simulate():
    
    for i in range(len(physicScene.balls)):
        
        physicScene.balls[i].simulate(physicScene.dt, physicScene.gravity)
        
        
        for j in range(i + 1, len(physicScene.balls)):
            handle_ball_collision(physicScene.balls[i], physicScene.balls[j])

        handle_wall_collision(physicScene.balls[i], physicScene.worldSize)


#mouse click to hit a ball
def handle_mouse_click(mouse_x, mouse_y):
    sim_x = screen_to_sim_x(mouse_x)
    sim_y = screen_to_sim_y(mouse_y)
    
    # Check if click is on any ball
    for ball in physicScene.balls:
        dx = ball.pos.x - sim_x
        dy = ball.pos.y - sim_y
        distance = math.sqrt(dx**2 + dy**2)

        if distance <= ball.radius:
            ball.vel.y += 15.0  
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
