import pygame
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import sys
import os 
# --- CONFIG ---
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
CAR_SIZE_X = 50
CAR_SIZE_Y = 30
BORDER_COLOR = (0, 0, 0, 255) # Black is Lava/Death
REWARD_COLOR = (255, 30, 30, 255) # Reward (Red)

# --- PYTORCH BRAIN (The Model) ---
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        pred = self.model(state)
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()


class Car:
    def __init__(self):
        self.sprite = pygame.image.load('car.png').convert() 
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
        self.rotated_sprite = self.sprite 
        self.position = [10, 10] # Starting Position
        self.angle = 270
        self.speed = 0
        self.center = [self.position[0] + CAR_SIZE_X / 2, self.position[1] + CAR_SIZE_Y / 2]
        self.radars = [] 
        self.alive = True
        self.on_red_line = False

    def draw(self, screen):
        new_rect = self.rotated_sprite.get_rect()
        
        new_rect.center = self.center
        
        screen.blit(self.rotated_sprite, new_rect.topleft)
        
        self.draw_radar(screen)

    def draw_radar(self, screen):
        for radar in self.radars:
            position = radar[0]
            pygame.draw.line(screen, (0, 255, 0), self.center, position, 1)
            pygame.draw.circle(screen, (0, 255, 0), position, 5)

    def check_collision(self, map_sprite):
        self.alive = True
        for point in self.corners:
            px = int(point[0])
            py = int(point[1])

            # 1. Safety Check: Is the point inside the map dimensions?
            if (px < 0 or px >= map_sprite.get_width() or 
                py < 0 or py >= map_sprite.get_height()):
                self.alive = False # Off-screen = Crash
                break

            # 2. Color Check: Is the point touching a wall?
            try:
                if map_sprite.get_at((px, py)) == BORDER_COLOR:
                    self.alive = False
                    break
            except IndexError:
                # Final safety net
                self.alive = False
                break

    def check_radar(self, degree, map_sprite):
        length = 0
        x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
        y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        while length < 300:
            # SAFETY CHECK: Stop if coordinates are outside the map image
            if (x < 0 or x >= map_sprite.get_width() or 
                y < 0 or y >= map_sprite.get_height()):
                break
            
            # CHECK COLOR: Wrap in try-except just in case
            try:
                if map_sprite.get_at((x, y)) == BORDER_COLOR:
                    break
            except IndexError:
                break

            length += 1
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        dist = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
        self.radars.append([(x, y), dist])

    def update(self, map_sprite, action=None):
        # speed
        self.speed = 5

        # 1. HANDLE AI INPUT (If action is provided)
        if action is not None:
            # Action logic: [Straight, Right, Left]
            if np.array_equal(action, [0, 1, 0]):
                self.angle -= 5 # Turn Right
            elif np.array_equal(action, [0, 0, 1]):
                self.angle += 5 # Turn Left
        
        # 2. Movement Logic (Same as before)
        self.position[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        self.position[1] += math.sin(math.radians(360 - self.angle)) * self.speed
        
        # ... (Keep the rest of your corner calculation and radar logic here) ...
        # (Copy the center calculation, corners, radars, collision from your previous code)
        self.center = [int(self.position[0]) + CAR_SIZE_X / 2, int(self.position[1]) + CAR_SIZE_Y / 2]
        
        # ... Recalculate corners ...
        left_top = [self.center[0] - CAR_SIZE_X/2, self.center[1] - CAR_SIZE_Y/2]
        right_top = [self.center[0] + CAR_SIZE_X/2, self.center[1] - CAR_SIZE_Y/2]
        left_bottom = [self.center[0] - CAR_SIZE_X/2, self.center[1] + CAR_SIZE_Y/2]
        right_bottom = [self.center[0] + CAR_SIZE_X/2, self.center[1] + CAR_SIZE_Y/2]

        self.corners = [
            self.rotate_point(left_top, self.center, self.angle),
            self.rotate_point(right_top, self.center, self.angle),
            self.rotate_point(left_bottom, self.center, self.angle),
            self.rotate_point(right_bottom, self.center, self.angle)
        ]

        self.radars.clear()
        for d in range(-90, 120, 45): 
            self.check_radar(d, map_sprite)
        
        self.check_collision(map_sprite)
        self.rotated_sprite = pygame.transform.rotate(self.sprite, self.angle)
        
    def rotate_point(self, point, center, angle):
        angle_rad = math.radians(360 - angle) # Convert to radians
        # Standard rotation matrix formula
        ox, oy = center
        px, py = point

        qx = ox + math.cos(angle_rad) * (px - ox) - math.sin(angle_rad) * (py - oy)
        qy = oy + math.sin(angle_rad) * (px - ox) + math.cos(angle_rad) * (py - oy)
        return [int(qx), int(qy)]


def train():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    map_sprite = pygame.image.load('track2.png').convert()
    map_sprite = pygame.transform.scale(map_sprite, (SCREEN_WIDTH, SCREEN_HEIGHT))
    font = pygame.font.SysFont("Arial", 25)

    # Init AI
    MAX_MEMORY = 100_000
    BATCH_SIZE = 1000
    LR = 0.001

    model = Linear_QNet(5, 256, 3) 
    trainer = QTrainer(model, lr=LR, gamma=0.9)
    memory = deque(maxlen=MAX_MEMORY)
    
    # --- NEW: LOADING LOGIC ---
    model_path = "model.pth"
    if os.path.exists(model_path):
        print("Loading saved model...")
        model.load_state_dict(torch.load(model_path))
        model.eval() # Set to evaluation mode
    else:
        print("No saved model found. Starting fresh.")
    # --------------------------

    car = Car()
    # epsilon = 0 # OLD WAY
    n_games = 0
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()

        state_old = [r[1] / 300 for r in car.radars] 
        if len(state_old) == 0: state_old = [0,0,0,0,0]

        # --- UPDATED: EPSILON LOGIC ---
        # If we loaded a model, we want less randomness (Exploration)
        # and more prediction (Exploitation).
        epsilon = 80 - n_games 
        
        
        final_move = [0,0,0]
        if random.randint(0, 200) < epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state_old, dtype=torch.float)
            prediction = model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        car.update(map_sprite, final_move)
        
        try:
            # Get integer coordinates of car center
            cx, cy = int(car.center[0]), int(car.center[1])
            
            # Check pixel color under the car
            pixel_color = map_sprite.get_at((cx, cy))
            
            if pixel_color == REWARD_COLOR:
                if not car.on_red_line: # Only reward if we weren't already touching it
                    reward = 100 # BIG REWARD
                    print(f"Hit Red Line! Reward +100")
                    car.on_red_line = True # Flag: "I am currently touching red"
            else:
                car.on_red_line = False # Reset flag when we drive off the line
                
        except IndexError:
            pass # Ignore if car center is slightly off-screen
        reward = 0
        done = False
        
        if not car.alive:
            reward = -100
            done = True
            car = Car() 
            n_games += 1
            
            # --- NEW: SAVING LOGIC ---
            # Save the model every time a game ends
            torch.save(model.state_dict(), model_path)
            print(f"Game {n_games} Saved!")
            # -------------------------
            
        else:
            reward = 10 

        state_new = [r[1] / 300 for r in car.radars]
        if len(state_new) == 0: state_new = [0,0,0,0,0]
        
        trainer.train_step(state_old, final_move, reward, state_new, done)
        memory.append((state_old, final_move, reward, state_new, done))

        if done and len(memory) > BATCH_SIZE:
            mini_sample = random.sample(memory, BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*mini_sample)
            trainer.train_step(states, actions, rewards, next_states, dones)

        screen.blit(map_sprite, (0, 0))
        car.draw(screen)
        
        current_inputs = [r[1] / 300 for r in car.radars]
        if len(current_inputs) == 5:
             draw_neural_net(screen, model, current_inputs)
        # Black Text
        text = font.render(f"Generation: {n_games}", True, (0, 0, 0))
        screen.blit(text, (10, 10))
        pygame.display.flip()
        
        #clock.tick(30)


def draw_neural_net(screen, model, car_inputs):
    # Coordinates
    start_x, start_y = 900, 50 # Top Right corner
    layer_gap = 150
    node_gap = 40
    
    # 1. Get the weights from the first layer (Input -> Hidden)
    # We simplify by averaging the 256 hidden neurons into just 1 "Average" connection strength
    # to keep the drawing clean.
    weights = model.linear1.weight.data.numpy() # Shape: [256, 5]
    
    # Inputs (5 Nodes)
    input_nodes = []
    for i in range(5):
        x = start_x
        y = start_y + i * node_gap
        input_nodes.append((x, y))
        
        # Color: Green if sensor sees far, Red if close
        value = car_inputs[i] # 0 to 1
        color = (int(255 * (1-value)), int(255 * value), 0)
        pygame.draw.circle(screen, color, (x, y), 10)
        
    output_nodes = []
    labels = ["R", "S", "L"] 
    for i in range(3):
        x = start_x + layer_gap
        y = start_y + i * node_gap + 40
        output_nodes.append((x, y))
        pygame.draw.circle(screen, (255, 255, 255), (x, y), 10)
        
        font = pygame.font.SysFont("Arial", 12)
        text = font.render(labels[i], True, (0,0,0))
        screen.blit(text, (x-4, y-8))


    for i in range(5): # Inputs
        for j in range(3): # Outputs
            # Draw a faint gray line connecting them
            pygame.draw.line(screen, (50, 50, 50), input_nodes[i], output_nodes[j], 1)
            
            # If input is high, make the line glow
            if car_inputs[i] > 0.5:
                 pygame.draw.line(screen, (0, 255, 0), input_nodes[i], output_nodes[j], 2)
# --- GAME LOOP ---
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()
car = Car()
map_sprite = pygame.image.load('track2.png').convert() 
map_sprite = pygame.transform.scale(map_sprite, (SCREEN_WIDTH, SCREEN_HEIGHT))

if __name__ == "__main__":
    train()