from flask import Flask, jsonify, request
import numpy as np
import torch
import os

# --- Import Core Project Files ---
# We are importing the Pygame-free logic and the AI model structure
from game import SnakeGameAI, Direction, Point, BLOCK_SIZE
from model import Linear_QNet 

# --- FLASK SETUP ---
app = Flask(__name__)

# --- AI AGENT CONFIGURATION ---
# IMPORTANT: These must match the settings used to train your model!
INPUT_SIZE = 11
HIDDEN_SIZE = 256
OUTPUT_SIZE = 3
MODEL_PATH = './model/model.pth' # Ensure your trained model is here

# --- AI MODEL LOADING ---
try:
    model = Linear_QNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval() # Set the model to evaluation mode (no training)
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}")
    print("Please ensure you have trained the agent and the 'model/model.pth' file exists.")
    exit()

# Initialize the persistent game state 
game = SnakeGameAI()


# --- HELPER FUNCTION: Get State (Needed by the server to decide the move) ---
def get_state(current_game):
    """Calculates the 11-dimensional state vector for the neural network."""
    head = current_game.snake[0]
    # Define points around the head to check for danger
    point_l = Point(head.x - BLOCK_SIZE, head.y)
    point_r = Point(head.x + BLOCK_SIZE, head.y)
    point_u = Point(head.x, head.y - BLOCK_SIZE)
    point_d = Point(head.x, head.y + BLOCK_SIZE)
    
    # Current direction
    dir_l, dir_r, dir_u, dir_d = (
        current_game.direction == Direction.LEFT,
        current_game.direction == Direction.RIGHT,
        current_game.direction == Direction.UP,
        current_game.direction == Direction.DOWN
    )

    state = [
        # Danger straight, right, left (logic copied from agent.py)
        (dir_r and current_game.is_collision(point_r)) or (dir_l and current_game.is_collision(point_l)) or (dir_u and current_game.is_collision(point_u)) or (dir_d and current_game.is_collision(point_d)),
        (dir_u and current_game.is_collision(point_r)) or (dir_d and current_game.is_collision(point_l)) or (dir_l and current_game.is_collision(point_u)) or (dir_r and current_game.is_collision(point_d)),
        (dir_d and current_game.is_collision(point_r)) or (dir_u and current_game.is_collision(point_l)) or (dir_r and current_game.is_collision(point_u)) or (dir_l and current_game.is_collision(point_d)),
        
        # Move direction
        dir_l, dir_r, dir_u, dir_d,
        
        # Food location 
        current_game.food.x < current_game.head.x,  # food left
        current_game.food.x > current_game.head.x,  # food right
        current_game.food.y < current_game.head.y,  # food up
        current_game.food.y > current_game.head.y   # food down
    ]
    return np.array(state, dtype=int)


# --- API ENDPOINT: Runs one AI step and returns the game state ---
@app.route('/api/step', methods=['GET'])
def play_ai_step():
    """Calculates the AI's next move, executes it, and returns the new state."""
    global game 

    # 1. Get current state and convert to PyTorch tensor
    state_old = get_state(game)
    state0 = torch.tensor(state_old, dtype=torch.float)

    # 2. Get AI move (Exploitation: no randomness)
    with torch.no_grad():
        prediction = model(state0)
    
    move_idx = torch.argmax(prediction).item()
    final_move = [0, 0, 0] # [straight, right, left]
    final_move[move_idx] = 1

    # 3. Perform move and get new state
    reward, done, score = game.play_step(final_move)
    
    if done:
        # If game over, reset the backend game state immediately
        game.reset()
    
    # 4. Prepare data as a JSON dictionary
    game_data = {
        # Convert snake Points to simple lists for JSON transport
        'snake': [{'x': p.x, 'y': p.y} for p in game.snake],
        'food': {'x': game.food.x, 'y': game.food.y},
        'score': score,
        'game_over': done,
        'w': game.w,
        'h': game.h,
        'block_size': BLOCK_SIZE
    }
    
    # Send JSON data back to the browser
    return jsonify(game_data)


# --- INDEX ENDPOINT: Serves the HTML/JavaScript UI ---
@app.route('/')
def index():
    """Serves the simple HTML page with the JavaScript logic to draw the snake."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Snake RL Agent Live Demo</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                text-align: center; 
                background-color: #282c34; 
                color: #61dafb;
            }
            #game-container { 
                display: inline-block; 
                border: 2px solid #61dafb; 
                margin-top: 20px; 
            }
            #score-board { 
                margin-bottom: 10px; 
                font-size: 2em; 
                color: white;
            }
            h1 { margin-bottom: 5px; }
            p { color: #aaa; }
        </style>
    </head>
    <body>
        <h1>Deep Q-Learning Snake</h1>
        <div id="score-board">Score: <span id="current-score">0</span></div>
        <div id="game-container">
            <canvas id="snakeCanvas"></canvas>
        </div>
        <p>AI running on a remote server. Refresh the page to restart the game.</p>

        <script>
            const canvas = document.getElementById('snakeCanvas');
            const ctx = canvas.getContext('2d');
            let gameSpeed = 100; // Default speed (10 steps per second)

            function drawGame(data) {
                // Set canvas dimensions
                canvas.width = data.w;
                canvas.height = data.h;

                // Clear canvas (Background: Black)
                ctx.fillStyle = '#000000'; 
                ctx.fillRect(0, 0, data.w, data.h);

                const BLOCK = data.block_size;

                // Draw Snake
                data.snake.forEach((pt) => {
                    ctx.fillStyle = '#0000FF'; // BLUE1
                    ctx.fillRect(pt.x, pt.y, BLOCK, BLOCK);
                    ctx.fillStyle = '#0064FF'; // BLUE2 (inner highlight)
                    ctx.fillRect(pt.x + 4, pt.y + 4, BLOCK - 8, BLOCK - 8);
                });

                // Draw Food (RED)
                ctx.fillStyle = '#FF0000'; 
                ctx.fillRect(data.food.x, data.food.y, BLOCK, BLOCK);

                // Update Score
                document.getElementById('current-score').textContent = data.score;
            }

            // This function continuously asks the server for the next game state
            function fetchAndStep() {
                fetch('/api/step')
                    .then(response => response.json())
                    .then(data => {
                        drawGame(data);

                        if (data.game_over) {
                            // When game is over, slow down the reset for visual effect
                            gameSpeed = 500; 
                        } else {
                            gameSpeed = 100; // Normal running speed
                        }
                        
                        // Loop: call the next step after a short delay
                        setTimeout(fetchAndStep, gameSpeed);
                    })
                    .catch(error => {
                        console.error('Error fetching game data:', error);
                        document.getElementById('score-board').textContent = 'CONNECTION ERROR: Cannot reach server.';
                    });
            }

            // Start the continuous game loop
            fetchAndStep();
        </script>
    </body>
    </html>
    """

if __name__ == '__main__':
    # Running locally for testing
    # In a real deployment, the web server (Gunicorn) will handle this.
    app.run(host='0.0.0.0', port=5000)