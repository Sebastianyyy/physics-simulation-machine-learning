from __future__ import annotations
from dataclasses import dataclass
import math
import time
from vpython import *


@dataclass
class Box:
    x: float
    y: float
    z : float
    vx: float
    vy: float
    vz: float
    size: float  # half-size
    mass : float
    color: str
    collision_timer: float = 0.0 
    
    def __post_init__(self):
        self.mass = (self.size * 2) ** 3  # volume of cube
        self.color = "blue"
        
    def left(self):
        return self.x - self.size

    def right(self):
        return self.x + self.size
    
    def top(self):
        return self.y + self.size
    
    def bottom(self):
        return self.y - self.size
    def front(self):
        return self.z + self.size
    def back(self):
        return self.z - self.size
    
    def update(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.z += self.vz * dt
        
        # Decrease collision timer
        if self.collision_timer > 0:
            self.collision_timer -= dt
    
    def mark_collision(self, duration=0.01):
        """Mark this box as colliding for a duration"""
        self.collision_timer = duration
    
    def is_colliding(self):
        """Check if box is currently marked as colliding"""
        return self.collision_timer > 0

    @staticmethod
    def boxes_collide(b1: Box, b2: Box) -> bool:
        """
        Check if bboxes colldie

        Args:
            b1 (Box): First bbox
            b2 (Box): Second bbox

        Returns:
            bool: True if bboxes collide, False otherwise
        """
        return (b1.left() < b2.right() and b1.right() > b2.left() and
                b1.bottom() < b2.top() and b1.top() > b2.bottom() and
                b1.back() < b2.front() and b1.front() > b2.back())



def sweet_and_prune(vboxes: list[tuple[Box, box]]):
    """Sweep and prune collision detection algorithm."""
    # Sort by left edge on x-axis
    sorted_boxes = sorted(vboxes, key=lambda pair: pair[0].left())
    
    # Check collisions with early termination
    for i, (b1, vb1) in enumerate(sorted_boxes):
        for j in range(i + 1, len(sorted_boxes)):
            b2, vb2 = sorted_boxes[j]
            if b2.left() > b1.right():
                break  # No more possible collisions along x-axis
            if Box.boxes_collide(b1, b2):
                b1.mark_collision()
                b2.mark_collision()
    # Update colors based on collision timer
    for b, vb in vboxes:
        if b.is_colliding():
            vb.color = color.red
        else:
            vb.color = color.blue

def expandBits(v):
    """Expands a 10-bit integer into 30 bits by inserting 2 zeros after each bit."""
    v = (v * 0x00010001) & 0xFF0000FF
    v = (v * 0x00000101) & 0x0F00F00F
    v = (v * 0x00000011) & 0xC30C30C3
    v = (v * 0x00000005) & 0x49249249
    return v

def calculate_morton_code(x, y, z, world_size):
    """Calculate 3D Morton code for spatial hashing."""
    # Normalize coordinates to [0,1] range based on world size
    def normalize_coord(val):
        return (val + world_size / 2) / world_size
    
    x = normalize_coord(x)
    y = normalize_coord(y)
    z = normalize_coord(z)
    
    # Clamp to ensure they're in [0,1]
    x = min(max(x, 0.0), 1.0)
    y = min(max(y, 0.0), 1.0)
    z = min(max(z, 0.0), 1.0)
    
    # Scale to range [0, 1023] for 10-bit encoding
    x = min(int(x * 1023), 1023)
    y = min(int(y * 1023), 1023)
    z = min(int(z * 1023), 1023)
    
    # Insert zeros between bits (3D Morton code)
    xx = expandBits(x)
    yy = expandBits(y)
    zz = expandBits(z)
    
    # Interleave the bits
    return xx | (yy << 1) | (zz << 2)



@dataclass
class AABB:
    """Axis-Aligned Bounding Box"""
    min_x: float = float('inf')
    min_y: float = float('inf')
    min_z: float = float('inf')
    max_x: float = float('-inf')
    max_y: float = float('-inf')
    max_z: float = float('-inf')
    
    def intersects(self, other: AABB) -> bool:
        """Check if this AABB intersects with another AABB"""
        return (
            self.min_x <= other.max_x and self.max_x >= other.min_x and
            self.min_y <= other.max_y and self.max_y >= other.min_y and
            self.min_z <= other.max_z and self.max_z >= other.min_z
        )
    
    @staticmethod
    def from_box(b: Box) -> AABB:
        """Create AABB from Box"""
        return AABB(b.left(), b.bottom(), b.back(), b.right(), b.top(), b.front())

class BVHNode:
    """Bounding Volume Hierarchy Node"""
    def __init__(self):
        self.left: BVHNode | None = None
        self.right: BVHNode | None = None
        self.box_id: int = -1  # Only leaf nodes have valid box_ids
        self.aabb: AABB = AABB()
    
    def is_leaf(self) -> bool:
        return self.box_id != -1

def get_split_pos(begin: int, end: int) -> int:
    """Simple middle split strategy"""
    return (begin + end) // 2

def create_leaf(box_id: int, box: Box) -> BVHNode:
    """Create a leaf node for a single box"""
    node = BVHNode()
    node.box_id = box_id
    node.aabb = AABB.from_box(box)
    return node

def create_subtree(sorted_list: list[dict], begin: int, end: int, boxes: list[Box]) -> BVHNode:
    """Recursively create BVH subtree"""
    if begin == end:
        return create_leaf(sorted_list[begin]['id'], boxes[sorted_list[begin]['id']])
    else:
        m = get_split_pos(begin, end)
        node = BVHNode()
        
        node.left = create_subtree(sorted_list, begin, m, boxes)
        node.right = create_subtree(sorted_list, m + 1, end, boxes)
        
        # Update node's AABB to encompass children's AABBs
        node.aabb.min_x = min(node.left.aabb.min_x, node.right.aabb.min_x)
        node.aabb.min_y = min(node.left.aabb.min_y, node.right.aabb.min_y)
        node.aabb.min_z = min(node.left.aabb.min_z, node.right.aabb.min_z)
        
        node.aabb.max_x = max(node.left.aabb.max_x, node.right.aabb.max_x)
        node.aabb.max_y = max(node.left.aabb.max_y, node.right.aabb.max_y)
        node.aabb.max_z = max(node.left.aabb.max_z, node.right.aabb.max_z)
        
        return node

def create_bvh_tree(boxes: list[Box], world_size: float) -> BVHNode:
    """Create BVH tree from list of boxes"""
    # Create list of box IDs with their Morton codes
    sorted_list = []
    for i, box in enumerate(boxes):
        morton_code = calculate_morton_code(box.x, box.y, box.z, world_size)
        sorted_list.append({'id': i, 'morton_code': morton_code})
    
    # Sort by Morton code for spatial locality
    sorted_list.sort(key=lambda x: x['morton_code'])
    
    # Create the BVH tree recursively
    return create_subtree(sorted_list, 0, len(sorted_list) - 1, boxes)

def find_collisions_bvh(box_id: int, box: Box, node: BVHNode, boxes: list[Box], 
                        collisions: list[tuple[int, int]], check_count: dict):
    """Find collisions between a box and the BVH tree"""
    check_count['value'] += 1
    
    # If this box's AABB doesn't intersect with the node's AABB, return
    box_aabb = AABB.from_box(box)
    if not box_aabb.intersects(node.aabb):
        return
    
    # If this is a leaf node
    if node.is_leaf():
        # Don't check collisions with self
        if node.box_id != box_id:
            # Check for actual collision between boxes
            if Box.boxes_collide(box, boxes[node.box_id]):
                collisions.append((box_id, node.box_id))
        return
    
    # Recurse through children
    find_collisions_bvh(box_id, box, node.left, boxes, collisions, check_count)
    find_collisions_bvh(box_id, box, node.right, boxes, collisions, check_count)

def check_collisions_bvh(boxes: list[Box], bvh_root: BVHNode) -> dict:
    """Check all collisions using BVH"""
    collisions = []
    check_count = {'value': 0}
    
    for i, box in enumerate(boxes):
        find_collisions_bvh(i, box, bvh_root, boxes, collisions, check_count)
    
    return {'collisions': collisions, 'check_count': check_count['value']}

def bvh_collision_detection(vboxes: list[tuple[Box, box]], world_size: float):
    """BVH-based collision detection for visualization"""
    # Extract boxes
    boxes = [b for b, vb in vboxes]
    
    # Build BVH tree
    bvh_root = create_bvh_tree(boxes, world_size)
    
    # Find collisions
    result = check_collisions_bvh(boxes, bvh_root)
    
    # Mark colliding boxes with timer
    for box_id_a, box_id_b in result['collisions']:
        boxes[box_id_a].mark_collision()
        boxes[box_id_b].mark_collision()
    
    # Update colors based on collision
    for b, vb in vboxes:
        if b.is_colliding():
            vb.color = color.red
        else:
            vb.color = color.blue
    
    return result['check_count']

def brute_force(vboxes: list[tuple[Box, box]]):
    """Brute force collision detection - checks all pairs"""
    # Check all pairs for collisions
    for i, (b1, vb1) in enumerate(vboxes):
        for j, (b2, vb2) in enumerate(vboxes):
            if i < j and Box.boxes_collide(b1, b2): 
                b1.mark_collision()
                b2.mark_collision()
    
    # Update colors based on collision
    for b, vb in vboxes:
        if b.is_colliding():
            vb.color = color.red
        else:
            vb.color = color.blue


def visualize_simulation(boxes: list[Box], box_size=1000, dt=0.01):
    """Real-time interactive 3D visualization"""
    
    scene.title = "Box Simulation - FPS: 0"
    scene.width = 1200
    scene.height = 800
    scene.range = box_size * 0.8
    scene.center = vector(box_size/2, box_size/2, box_size/2)
    scene.forward = vector(-1, -1, -1)
    scene.userpan = True
    scene.userzoom = True
    scene.userspin = True
    
    # Create transparent box boundaries
    box(pos=vector(box_size/2, box_size/2, box_size/2), 
        size=vector(box_size, box_size, box_size), 
        opacity=0.1, color=color.white)
    
    # Create floor
    box(pos=vector(box_size/2, 0, box_size/2), 
        size=vector(box_size, 2, box_size), 
        color=color.gray(0.7))
    
    # Create VPython boxes matching your Box objects
    vboxes = []
    for b in boxes:
        c = color.red if b.color == "red" else (color.green if b.color == "green" else color.blue)
        vb = box(pos=vector(b.x, b.y, b.z), 
                 size=vector(b.size*2, b.size*2, b.size*2),
                 color=c)
        vboxes.append((b, vb))
    
    # FPS tracking
    frame_count = 0
    fps_update_interval = 10  # Update FPS display every 10 frames
    last_time = time.perf_counter()
    
    # Choose collision detection method
    USE_BRUTE_FORCE = False
    USE_SWEEP_PRUNE = True
    USE_BVH = True
    
    if USE_BRUTE_FORCE:
        collision_method = "Brute Force"
    elif USE_SWEEP_PRUNE:
        collision_method = "Sweep & Prune"
    else:
        collision_method = "BVH"
    
    # Animation loop
    while True:
        rate(100) 
                
        for b, vb in vboxes:
            # Update physics
            b.update(dt)
            
            # Handle wall collisions (simple bounce)
            if b.left() <= 0 or b.right() >= box_size:
                b.vx *= -0.9  # bounce with energy loss
            if b.bottom() <= 0 or b.top() >= box_size:
                b.vy *= -0.9
            if b.back() <= 0 or b.front() >= box_size:
                b.vz *= -0.9
            
            # Keep box in bounds
            b.x = max(b.size, min(box_size - b.size, b.x))
            b.y = max(b.size, min(box_size - b.size, b.y))
            b.z = max(b.size, min(box_size - b.size, b.z))
            
            # Update visual position
            vb.pos = vector(b.x, b.y, b.z)
        
        # Collision detection - choose method
        collision_start = time.perf_counter()
        if USE_BRUTE_FORCE:
            brute_force(vboxes)
        elif USE_SWEEP_PRUNE:
            sweet_and_prune(vboxes)
        else:
            bvh_collision_detection(vboxes, box_size)
        collision_time = (time.perf_counter() - collision_start) * 1000  # ms
        
        frame_count += 1
        
        # Update FPS display
        if frame_count % fps_update_interval == 0:
            current_time = time.perf_counter()
            elapsed = current_time - last_time
            fps = fps_update_interval / elapsed
            last_time = current_time
            
            # Update scene title with FPS and collision time
            scene.title = f"Box Simulation | FPS: {fps:.1f} | {collision_method}: {collision_time:.2f}ms | Boxes: {len(vboxes)}"

if __name__ == "__main__":
    import random
    
    NUM_BOXES = 300
    
    # Auto-compute grid dimensions (as close to cube as possible)
    grid_dim = int(round(NUM_BOXES ** (1/3)))  # cube root for cubic grid
    
    # Adjust to get closest number to NUM_BOXES
    actual_boxes = grid_dim ** 3
    if actual_boxes < NUM_BOXES:
        # Try adding one dimension
        grid_x = grid_dim + 1
        grid_y = grid_dim
        grid_z = grid_dim
        if grid_x * grid_y * grid_z < NUM_BOXES:
            grid_y = grid_dim + 1
        if grid_x * grid_y * grid_z < NUM_BOXES:
            grid_z = grid_dim + 1
    else:
        grid_x = grid_y = grid_z = grid_dim
    
    # Auto-compute other parameters based on number of boxes
    box_size = 5000  # Fixed container size
    box_half_size = max(10, box_size // (grid_dim * 6))  # Auto size based on grid
    box_spacing = (box_size * 0.8) / grid_dim  # Spacing to fit in container
    velocity_magnitude = 100  # Constant velocity
    
    # Create boxes in a 3D grid
    boxes = []
    for i in range(grid_x):
        for j in range(grid_y):
            for k in range(grid_z):
                if len(boxes) >= NUM_BOXES:  # Stop when we reach desired number
                    break
                    
                # Position in grid
                x = box_size * 0.1 + i * box_spacing
                y = box_size * 0.1 + j * box_spacing
                z = box_size * 0.1 + k * box_spacing
                
                # Random velocity direction (normalized then scaled)
                vx = random.uniform(-1, 1)
                vy = random.uniform(-1, 1)
                vz = random.uniform(-1, 1)
                
                # Normalize and scale to velocity_magnitude
                v_length = math.sqrt(vx**2 + vy**2 + vz**2)
                vx = (vx / v_length) * velocity_magnitude
                vy = (vy / v_length) * velocity_magnitude
                vz = (vz / v_length) * velocity_magnitude
                
                boxes.append(Box(x, y, z, vx, vy, vz, box_half_size, 0, "blue"))
            if len(boxes) >= NUM_BOXES:
                break
        if len(boxes) >= NUM_BOXES:
            break
    
    print(f"Created {len(boxes)} boxes in {grid_x}x{grid_y}x{grid_z} grid")
    print(f"Box size: {box_half_size*2}, Spacing: {box_spacing:.1f}, Container: {box_size}")
    
    # Uncomment to run performance comparison
    # Create vboxes for testing (without visualization)
    test_vboxes = [(b, None) for b in boxes]  # Mock vboxes for testing
    # compare_performance(test_vboxes, box_size, iterations=50)
    
    visualize_simulation(boxes, box_size=box_size, dt=0.01)

    # sweet_and_prune()