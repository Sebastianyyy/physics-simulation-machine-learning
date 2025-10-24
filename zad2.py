from __future__ import annotations
from dataclasses import dataclass

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

@dataclass
class PhysicsScene:
    gravity: Vector2
    dt: float
    numSteps: int
    bead: list
    wireCenter: Vector2
    wireRadius: float
    wireVelocity: Vector2
    wireMass: float
    



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

class Bead:
    def __init__(self, radius: float, mass: float, pos: Vector2):
        self.radius = radius
        self.mass = mass
        self.pos = pos.clone()
        self.prevPos = pos.clone()
        self.vel = Vector2()
        self.force = Vector2()

    def startStep(self, dt: float, gravity: Vector2):
        self.vel.add(gravity, dt)
        self.prevPos.set(self.pos)
        self.pos.add(self.vel, dt)
        
    def keepOnWire(self, center: Vector2, radius: float, dt: float):
        # Calculate direction from center to bead
        diri = Vector2()
        diri.substractVectors(self.pos, center)
        length = diri.length()
        if length == 0.0:
            self.force.set(Vector2(0.0, 0.0))
            return
        
        # Normalize direction (points from center toward bead)
        diri.scale(1.0 / length)
        
        # Calculate how much we need to correct position
        lambd = physicScene.wireRadius - length
        
        # Move bead back to wire
        self.pos.add(diri, lambd)
        
        # Calculate constraint force on the BEAD
        # The force points radially: inward if bead tries to leave, outward if it goes inside
        # lambd > 0 means bead is inside circle -> push outward (positive diri)
        # lambd < 0 means bead is outside circle -> pull inward (negative diri)
        # F = m * a / dt^2
        forceMagnitude = self.mass * lambd / (dt * dt)
        self.force.set(diri)
        self.force.scale(forceMagnitude)
        
    def endStep(self, dt: float):
        self.vel.substractVectors(self.pos, self.prevPos)
        self.vel.scale(1.0 / dt)
    
physicScene=PhysicsScene(
    gravity=Vector2(0.0, -10.0),
    dt=1.0 / 60.0,
    numSteps=100,
    wireCenter=Vector2(0.0, 0.0),
    wireRadius=0.0,
    bead=[],
    wireVelocity=Vector2(0.0, 0.0),
    wireMass=float('inf')
)


def setupScene():
    physicScene.bead=[]
    physicScene.wireCenter.x = sim_width / 2.0
    physicScene.wireCenter.y = sim_height / 2.0
    physicScene.wireRadius = sim_width * 0.1
    pos = Vector2(physicScene.wireCenter.x + physicScene.wireRadius, physicScene.wireCenter.y)
    
    numBeads=5
    mass = 1.0
    r=0.4
    angle = 0.0
    for i in range(numBeads):
        mass = math.pi * r * r
        pos = Vector2(physicScene.wireCenter.x + physicScene.wireRadius * math.cos(angle),
                      physicScene.wireCenter.y + physicScene.wireRadius * math.sin(angle))
        
        physicScene.bead.append(Bead(radius=r, mass=mass, pos=pos))
        angle+= math.pi / numBeads
        r = 0.05 + random.random() * 0.5
        
    
setupScene()
# Create multiple balls



# Colors
RED = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Clock for controlling frame rate
clock = pygame.time.Clock()


# Draw a scene
def draw():
    screen.fill(WHITE)
    pygame.draw.circle(
        screen,
        BLACK,
        (c_x(physicScene.wireCenter.x), c_y(physicScene.wireCenter.y)),
        int(c_scale * physicScene.wireRadius), 2
    )

    # Draw beads
    for bead in physicScene.bead:
        pygame.draw.circle(
            screen,
            RED,
            (c_x(bead.pos.x), c_y(bead.pos.y)),
            int(c_scale * bead.radius)
        )
    pygame.display.flip()


def handleBeadBeadCollisions(bead1, bead2):
    restitution = 1.0
    diri = Vector2()
    diri.substractVectors(bead2.pos, bead1.pos)
    d = diri.length()
    if (d==0 or d > bead1.radius + bead2.radius):
        return
    
    diri.scale(1.0/d)
    
    corr = (bead1.radius + bead2.radius - d) / 2.0
    bead1.pos.add(diri, -corr)
    bead2.pos.add(diri, corr)
    
    v1 = bead1.vel.dot(diri)
    v2 = bead2.vel.dot(diri)
    
    m1 = bead1.mass
    m2 = bead2.mass
    
    newv1 = (m1 * v1 + m2 * v2 - m2 * (v1-v2)* restitution) / (m1 + m2)
    newv2 = (m1 * v1 + m2 * v2 - m1 * (v2 - v1) * restitution) / (m1 + m2)
    
    bead1.vel.add(diri, newv1 - v1)
    bead2.vel.add(diri, newv2 - v2)
    


def simulate():

    sdt = physicScene.dt / physicScene.numSteps
    
    for step in range(physicScene.numSteps):
        #  Apply gravity and update positions
        for bead in physicScene.bead:
            bead.startStep(sdt, physicScene.gravity)
        
        # Enforce wire constraint and calculate forces
        totalForceOnWire = Vector2(0.0, 0.0)
        for bead in physicScene.bead:
            bead.keepOnWire(physicScene.wireCenter, physicScene.wireRadius, sdt)
            # By Newton's 3rd law: force on wire = -force on bead
            # If bead is pulled inward (negative force), wire is pushed outward (positive)
            totalForceOnWire.substract(bead.force)
        
        # Update wire center if it has finite mass
            # F = ma => a = F/m
        wireAcceleration = totalForceOnWire.clone()
        wireAcceleration.scale(1.0 / physicScene.wireMass)
        physicScene.wireVelocity.add(wireAcceleration, sdt)
        physicScene.wireCenter.add(physicScene.wireVelocity, sdt)
        
        # Update velocities
        for bead in physicScene.bead:
            bead.endStep(sdt)
        
        for i in range(len(physicScene.bead)):
            for j in range(i + 1, len(physicScene.bead)):
                handleBeadBeadCollisions(physicScene.bead[i], physicScene.bead[j])


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
        simulate()        
        draw()
        clock.tick(60)
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
