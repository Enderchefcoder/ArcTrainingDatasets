#!/usr/bin/env python3
"""
Comprehensive training pair generation system for ARC-AGI challenges.
Generates thousands of training pairs for grid sizes from 2x2 to 24x24.
"""

import json
import random
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import itertools
from collections import defaultdict

from src.models import Challenge, Example, GRID
from src.prompts.colors import color_map


class TransformationType(Enum):
    COLOR_CHANGE = "color_change"
    SHAPE_OPERATION = "shape_operation"
    PATTERN_RECOGNITION = "pattern_recognition"
    GRID_MANIPULATION = "grid_manipulation"
    MATHEMATICAL = "mathematical"
    GEOMETRIC = "geometric"


@dataclass
class GenerationConfig:
    """Configuration for training pair generation."""
    min_grid_size: int = 2
    max_grid_size: int = 24
    examples_per_challenge: int = 3
    total_challenges: int = 10000
    color_range: Tuple[int, int] = (0, 9)
    complexity_levels: List[str] = None
    
    def __post_init__(self):
        if self.complexity_levels is None:
            self.complexity_levels = ["simple", "medium", "complex"]


class TrainingPairGenerator:
    """Generates diverse training pairs for ARC-AGI challenges."""
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.generated_challenges = []
        self.transformation_types = list(TransformationType)
        
    def generate_random_grid(self, rows: int, cols: int, 
                           color_weights: Optional[Dict[int, float]] = None) -> GRID:
        """Generate a random grid with specified dimensions."""
        if color_weights is None:
            # Default: more black (0) and fewer colored cells
            color_weights = {0: 0.7, 1: 0.05, 2: 0.05, 3: 0.05, 4: 0.05, 
                           5: 0.02, 6: 0.02, 7: 0.02, 8: 0.02, 9: 0.02}
        
        colors = list(color_weights.keys())
        weights = list(color_weights.values())
        
        grid = []
        for _ in range(rows):
            row = np.random.choice(colors, size=cols, p=weights).tolist()
            grid.append(row)
        
        return grid
    
    def generate_color_change_challenge(self, grid_size: Tuple[int, int]) -> Challenge:
        """Generate a challenge involving color changes."""
        rows, cols = grid_size
        examples = []
        
        # Choose a transformation rule
        rule_type = random.choice([
            "replace_color", "conditional_color_change", "color_shift", 
            "color_inversion", "color_based_pattern"
        ])
        
        for i in range(self.config.examples_per_challenge):
            input_grid = self.generate_random_grid(rows, cols)
            output_grid = self.apply_color_transformation(input_grid, rule_type)
            
            examples.append(Example(
                input=input_grid,
                output=output_grid
            ))
        
        # Generate test case
        test_input = self.generate_random_grid(rows, cols)
        test_output = self.apply_color_transformation(test_input, rule_type)
        
        challenge_id = f"color_change_{rows}x{cols}_{random.randint(1000, 9999)}"
        return Challenge(
            id=challenge_id,
            train=examples,
            test=[Example(input=test_input, output=test_output)]
        )
    
    def apply_color_transformation(self, grid: GRID, rule_type: str) -> GRID:
        """Apply a color transformation rule to a grid."""
        grid = [row[:] for row in grid]  # Deep copy
        
        if rule_type == "replace_color":
            # Replace one specific color with another
            from_color = random.randint(1, 9)
            to_color = random.randint(1, 9)
            while to_color == from_color:
                to_color = random.randint(1, 9)
            
            for i in range(len(grid)):
                for j in range(len(grid[i])):
                    if grid[i][j] == from_color:
                        grid[i][j] = to_color
        
        elif rule_type == "conditional_color_change":
            # Change color based on position or surrounding colors
            for i in range(len(grid)):
                for j in range(len(grid[i])):
                    if grid[i][j] != 0:  # Not black
                        # Change based on position
                        if (i + j) % 2 == 0:
                            grid[i][j] = (grid[i][j] + 1) % 10
                        else:
                            grid[i][j] = (grid[i][j] - 1) % 10
        
        elif rule_type == "color_shift":
            # Shift all colors by a fixed amount
            shift = random.randint(1, 8)
            for i in range(len(grid)):
                for j in range(len(grid[i])):
                    if grid[i][j] != 0:
                        grid[i][j] = (grid[i][j] + shift) % 10
        
        elif rule_type == "color_inversion":
            # Invert colors (0 stays 0, others become 10 - color)
            for i in range(len(grid)):
                for j in range(len(grid[i])):
                    if grid[i][j] != 0:
                        grid[i][j] = 10 - grid[i][j]
        
        elif rule_type == "color_based_pattern":
            # Create patterns based on existing colors
            for i in range(len(grid)):
                for j in range(len(grid[i])):
                    if grid[i][j] != 0:
                        # Create a simple pattern based on the color
                        if grid[i][j] % 2 == 0:
                            grid[i][j] = 2  # Red for even colors
                        else:
                            grid[i][j] = 1  # Blue for odd colors
        
        return grid
    
    def generate_shape_operation_challenge(self, grid_size: Tuple[int, int]) -> Challenge:
        """Generate a challenge involving shape operations."""
        rows, cols = grid_size
        examples = []
        
        rule_type = random.choice([
            "rotate_shapes", "flip_shapes", "move_shapes", "scale_shapes",
            "extract_shapes", "fill_shapes"
        ])
        
        for i in range(self.config.examples_per_challenge):
            input_grid = self.generate_shape_grid(rows, cols)
            output_grid = self.apply_shape_transformation(input_grid, rule_type)
            
            examples.append(Example(
                input=input_grid,
                output=output_grid
            ))
        
        test_input = self.generate_shape_grid(rows, cols)
        test_output = self.apply_shape_transformation(test_input, rule_type)
        
        challenge_id = f"shape_op_{rows}x{cols}_{random.randint(1000, 9999)}"
        return Challenge(
            id=challenge_id,
            train=examples,
            test=[Example(input=test_input, output=test_output)]
        )
    
    def generate_shape_grid(self, rows: int, cols: int) -> GRID:
        """Generate a grid with recognizable shapes."""
        grid = [[0 for _ in range(cols)] for _ in range(rows)]
        
        # Add some geometric shapes
        shape_type = random.choice(["rectangle", "line", "cross", "triangle", "circle"])
        color = random.randint(1, 9)
        
        if shape_type == "rectangle":
            # Random rectangle
            x1, y1 = random.randint(0, rows-2), random.randint(0, cols-2)
            x2, y2 = random.randint(x1+1, rows-1), random.randint(y1+1, cols-1)
            for i in range(x1, x2+1):
                for j in range(y1, y2+1):
                    grid[i][j] = color
        
        elif shape_type == "line":
            # Horizontal or vertical line
            if random.choice([True, False]):  # Horizontal
                row = random.randint(0, rows-1)
                start_col = random.randint(0, cols-2)
                end_col = random.randint(start_col+1, cols-1)
                for j in range(start_col, end_col+1):
                    grid[row][j] = color
            else:  # Vertical
                col = random.randint(0, cols-1)
                start_row = random.randint(0, rows-2)
                end_row = random.randint(start_row+1, rows-1)
                for i in range(start_row, end_row+1):
                    grid[i][col] = color
        
        elif shape_type == "cross":
            # Cross pattern
            center_row, center_col = rows // 2, cols // 2
            for i in range(max(0, center_row-1), min(rows, center_row+2)):
                grid[i][center_col] = color
            for j in range(max(0, center_col-1), min(cols, center_col+2)):
                grid[center_row][j] = color
        
        return grid
    
    def apply_shape_transformation(self, grid: GRID, rule_type: str) -> GRID:
        """Apply a shape transformation rule."""
        grid = [row[:] for row in grid]  # Deep copy
        rows, cols = len(grid), len(grid[0])
        
        if rule_type == "rotate_shapes":
            # Rotate the entire grid 90 degrees clockwise
            return [[grid[rows-1-j][i] for j in range(rows)] for i in range(cols)]
        
        elif rule_type == "flip_shapes":
            # Flip horizontally
            return [row[::-1] for row in grid]
        
        elif rule_type == "move_shapes":
            # Move all non-zero elements by a fixed offset
            dx, dy = random.randint(-1, 1), random.randint(-1, 1)
            new_grid = [[0 for _ in range(cols)] for _ in range(rows)]
            for i in range(rows):
                for j in range(cols):
                    if grid[i][j] != 0:
                        new_i, new_j = i + dx, j + dy
                        if 0 <= new_i < rows and 0 <= new_j < cols:
                            new_grid[new_i][new_j] = grid[i][j]
            return new_grid
        
        elif rule_type == "scale_shapes":
            # Scale up shapes (simple version)
            new_rows, new_cols = rows * 2, cols * 2
            if new_rows <= 24 and new_cols <= 24:
                new_grid = [[0 for _ in range(new_cols)] for _ in range(new_rows)]
                for i in range(rows):
                    for j in range(cols):
                        if grid[i][j] != 0:
                            new_grid[i*2][j*2] = grid[i][j]
                            if i*2+1 < new_rows and j*2+1 < new_cols:
                                new_grid[i*2+1][j*2] = grid[i][j]
                                new_grid[i*2][j*2+1] = grid[i][j]
                                new_grid[i*2+1][j*2+1] = grid[i][j]
                return new_grid
        
        return grid
    
    def generate_pattern_recognition_challenge(self, grid_size: Tuple[int, int]) -> Challenge:
        """Generate a challenge involving pattern recognition."""
        rows, cols = grid_size
        examples = []
        
        rule_type = random.choice([
            "find_edges", "count_objects", "detect_symmetry", "find_contours",
            "pattern_matching", "feature_extraction"
        ])
        
        for i in range(self.config.examples_per_challenge):
            input_grid = self.generate_pattern_grid(rows, cols)
            output_grid = self.apply_pattern_transformation(input_grid, rule_type)
            
            examples.append(Example(
                input=input_grid,
                output=output_grid
            ))
        
        test_input = self.generate_pattern_grid(rows, cols)
        test_output = self.apply_pattern_transformation(test_input, rule_type)
        
        challenge_id = f"pattern_{rows}x{cols}_{random.randint(1000, 9999)}"
        return Challenge(
            id=challenge_id,
            train=examples,
            test=[Example(input=test_input, output=test_output)]
        )
    
    def generate_pattern_grid(self, rows: int, cols: int) -> GRID:
        """Generate a grid with complex patterns."""
        grid = [[0 for _ in range(cols)] for _ in range(rows)]
        
        pattern_type = random.choice(["checkerboard", "stripes", "spiral", "fractal", "noise"])
        
        if pattern_type == "checkerboard":
            for i in range(rows):
                for j in range(cols):
                    if (i + j) % 2 == 0:
                        grid[i][j] = random.randint(1, 9)
        
        elif pattern_type == "stripes":
            stripe_color = random.randint(1, 9)
            stripe_width = random.randint(1, 3)
            for i in range(rows):
                if (i // stripe_width) % 2 == 0:
                    for j in range(cols):
                        grid[i][j] = stripe_color
        
        elif pattern_type == "spiral":
            # Create a simple spiral pattern
            color = random.randint(1, 9)
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            x, y, d = 0, 0, 0
            visited = set()
            
            for _ in range(min(rows * cols, 20)):  # Limit spiral size
                if 0 <= x < rows and 0 <= y < cols and (x, y) not in visited:
                    grid[x][y] = color
                    visited.add((x, y))
                
                nx, ny = x + directions[d][0], y + directions[d][1]
                if (nx < 0 or nx >= rows or ny < 0 or ny >= cols or (nx, ny) in visited):
                    d = (d + 1) % 4
                    nx, ny = x + directions[d][0], y + directions[d][1]
                
                x, y = nx, ny
        
        return grid
    
    def apply_pattern_transformation(self, grid: GRID, rule_type: str) -> GRID:
        """Apply a pattern recognition transformation."""
        grid = [row[:] for row in grid]  # Deep copy
        rows, cols = len(grid), len(grid[0])
        
        if rule_type == "find_edges":
            # Simple edge detection
            new_grid = [[0 for _ in range(cols)] for _ in range(rows)]
            for i in range(1, rows-1):
                for j in range(1, cols-1):
                    if grid[i][j] != 0:
                        # Check if it's on an edge (adjacent to different color or black)
                        neighbors = [grid[i-1][j], grid[i+1][j], grid[i][j-1], grid[i][j+1]]
                        if any(n != grid[i][j] for n in neighbors):
                            new_grid[i][j] = 2  # Red for edges
                        else:
                            new_grid[i][j] = 1  # Blue for interior
        
        elif rule_type == "count_objects":
            # Count connected components and mark them
            visited = [[False for _ in range(cols)] for _ in range(rows)]
            component_id = 1
            
            def dfs(i, j, color):
                if (i < 0 or i >= rows or j < 0 or j >= cols or 
                    visited[i][j] or grid[i][j] != color):
                    return
                visited[i][j] = True
                new_grid[i][j] = component_id
                for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    dfs(i + di, j + dj, color)
            
            for i in range(rows):
                for j in range(cols):
                    if grid[i][j] != 0 and not visited[i][j]:
                        dfs(i, j, grid[i][j])
                        component_id = min(component_id + 1, 9)
        
        elif rule_type == "detect_symmetry":
            # Detect horizontal symmetry
            new_grid = [[0 for _ in range(cols)] for _ in range(rows)]
            for i in range(rows):
                for j in range(cols):
                    if grid[i][j] != 0:
                        # Check if symmetric
                        sym_j = cols - 1 - j
                        if sym_j >= 0 and grid[i][sym_j] == grid[i][j]:
                            new_grid[i][j] = 2  # Red for symmetric
                        else:
                            new_grid[i][j] = 1  # Blue for asymmetric
        
        return new_grid
    
    def generate_mathematical_challenge(self, grid_size: Tuple[int, int]) -> Challenge:
        """Generate a challenge involving mathematical operations."""
        rows, cols = grid_size
        examples = []
        
        rule_type = random.choice([
            "sum_neighbors", "multiply_by_position", "modulo_operation",
            "distance_transform", "gradient_calculation"
        ])
        
        for i in range(self.config.examples_per_challenge):
            input_grid = self.generate_math_grid(rows, cols)
            output_grid = self.apply_math_transformation(input_grid, rule_type)
            
            examples.append(Example(
                input=input_grid,
                output=output_grid
            ))
        
        test_input = self.generate_math_grid(rows, cols)
        test_output = self.apply_math_transformation(test_input, rule_type)
        
        challenge_id = f"math_{rows}x{cols}_{random.randint(1000, 9999)}"
        return Challenge(
            id=challenge_id,
            train=examples,
            test=[Example(input=test_input, output=test_output)]
        )
    
    def generate_math_grid(self, rows: int, cols: int) -> GRID:
        """Generate a grid suitable for mathematical operations."""
        grid = []
        for i in range(rows):
            row = []
            for j in range(cols):
                # Generate numbers that work well with mathematical operations
                value = random.randint(0, 5)  # Smaller range for math operations
                row.append(value)
            grid.append(row)
        return grid
    
    def apply_math_transformation(self, grid: GRID, rule_type: str) -> GRID:
        """Apply a mathematical transformation."""
        grid = [row[:] for row in grid]  # Deep copy
        rows, cols = len(grid), len(grid[0])
        new_grid = [[0 for _ in range(cols)] for _ in range(rows)]
        
        if rule_type == "sum_neighbors":
            for i in range(rows):
                for j in range(cols):
                    total = 0
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < rows and 0 <= nj < cols:
                                total += grid[ni][nj]
                    new_grid[i][j] = min(total % 10, 9)
        
        elif rule_type == "multiply_by_position":
            for i in range(rows):
                for j in range(cols):
                    new_grid[i][j] = (grid[i][j] * (i + j + 1)) % 10
        
        elif rule_type == "modulo_operation":
            mod_value = random.randint(2, 5)
            for i in range(rows):
                for j in range(cols):
                    new_grid[i][j] = grid[i][j] % mod_value
        
        return new_grid
    
    def generate_geometric_challenge(self, grid_size: Tuple[int, int]) -> Challenge:
        """Generate a challenge involving geometric operations."""
        rows, cols = grid_size
        examples = []
        
        rule_type = random.choice([
            "distance_from_center", "angle_calculation", "convex_hull",
            "voronoi_diagram", "delaunay_triangulation"
        ])
        
        for i in range(self.config.examples_per_challenge):
            input_grid = self.generate_geometric_grid(rows, cols)
            output_grid = self.apply_geometric_transformation(input_grid, rule_type)
            
            examples.append(Example(
                input=input_grid,
                output=output_grid
            ))
        
        test_input = self.generate_geometric_grid(rows, cols)
        test_output = self.apply_geometric_transformation(test_input, rule_type)
        
        challenge_id = f"geo_{rows}x{cols}_{random.randint(1000, 9999)}"
        return Challenge(
            id=challenge_id,
            train=examples,
            test=[Example(input=test_input, output=test_output)]
        )
    
    def generate_geometric_grid(self, rows: int, cols: int) -> GRID:
        """Generate a grid with geometric features."""
        grid = [[0 for _ in range(cols)] for _ in range(rows)]
        
        # Add some geometric points
        num_points = random.randint(2, min(5, rows * cols // 4))
        points = []
        for _ in range(num_points):
            x, y = random.randint(0, rows-1), random.randint(0, cols-1)
            color = random.randint(1, 9)
            grid[x][y] = color
            points.append((x, y, color))
        
        return grid
    
    def apply_geometric_transformation(self, grid: GRID, rule_type: str) -> GRID:
        """Apply a geometric transformation."""
        grid = [row[:] for row in grid]  # Deep copy
        rows, cols = len(grid), len(grid[0])
        new_grid = [[0 for _ in range(cols)] for _ in range(rows)]
        
        if rule_type == "distance_from_center":
            center_x, center_y = rows // 2, cols // 2
            for i in range(rows):
                for j in range(cols):
                    if grid[i][j] != 0:
                        distance = int(((i - center_x) ** 2 + (j - center_y) ** 2) ** 0.5)
                        new_grid[i][j] = min(distance % 10, 9)
        
        return new_grid
    
    def generate_challenge(self, grid_size: Tuple[int, int]) -> Challenge:
        """Generate a random challenge of the specified grid size."""
        transformation_type = random.choice(self.transformation_types)
        
        if transformation_type == TransformationType.COLOR_CHANGE:
            return self.generate_color_change_challenge(grid_size)
        elif transformation_type == TransformationType.SHAPE_OPERATION:
            return self.generate_shape_operation_challenge(grid_size)
        elif transformation_type == TransformationType.PATTERN_RECOGNITION:
            return self.generate_pattern_recognition_challenge(grid_size)
        elif transformation_type == TransformationType.MATHEMATICAL:
            return self.generate_mathematical_challenge(grid_size)
        elif transformation_type == TransformationType.GEOMETRIC:
            return self.generate_geometric_challenge(grid_size)
        else:
            return self.generate_color_change_challenge(grid_size)  # Default fallback
    
    def generate_all_challenges(self) -> List[Challenge]:
        """Generate all challenges according to the configuration."""
        challenges = []
        
        # Calculate challenges per grid size
        total_grid_sizes = self.config.max_grid_size - self.config.min_grid_size + 1
        challenges_per_size = self.config.total_challenges // total_grid_sizes
        
        print(f"Generating {self.config.total_challenges} challenges...")
        print(f"Grid sizes: {self.config.min_grid_size}x{self.config.min_grid_size} to {self.config.max_grid_size}x{self.config.max_grid_size}")
        print(f"Challenges per size: {challenges_per_size}")
        
        for size in range(self.config.min_grid_size, self.config.max_grid_size + 1):
            print(f"Generating challenges for {size}x{size} grids...")
            
            for i in range(challenges_per_size):
                try:
                    challenge = self.generate_challenge((size, size))
                    challenges.append(challenge)
                    
                    if (i + 1) % 100 == 0:
                        print(f"  Generated {i + 1}/{challenges_per_size} challenges for {size}x{size}")
                
                except Exception as e:
                    print(f"  Error generating challenge {i+1} for {size}x{size}: {e}")
                    continue
        
        # Add remaining challenges to reach total
        remaining = self.config.total_challenges - len(challenges)
        if remaining > 0:
            print(f"Adding {remaining} additional challenges...")
            for i in range(remaining):
                size = random.randint(self.config.min_grid_size, self.config.max_grid_size)
                try:
                    challenge = self.generate_challenge((size, size))
                    challenges.append(challenge)
                except Exception as e:
                    print(f"Error generating additional challenge {i+1}: {e}")
                    continue
        
        print(f"Generated {len(challenges)} total challenges")
        return challenges
    
    def save_challenges(self, challenges: List[Challenge], output_path: Path):
        """Save challenges to JSON file."""
        challenges_dict = {}
        for challenge in challenges:
            challenges_dict[challenge.id] = {
                "id": challenge.id,
                "train": [{"input": ex.input, "output": ex.output} for ex in challenge.train],
                "test": [{"input": ex.input, "output": ex.output} for ex in challenge.test]
            }
        
        with open(output_path, 'w') as f:
            json.dump(challenges_dict, f, indent=2)
        
        print(f"Saved {len(challenges)} challenges to {output_path}")


def main():
    """Main function to generate training pairs."""
    config = GenerationConfig(
        min_grid_size=2,
        max_grid_size=24,
        examples_per_challenge=3,
        total_challenges=10000,
        color_range=(0, 9)
    )
    
    generator = TrainingPairGenerator(config)
    challenges = generator.generate_all_challenges()
    
    # Save challenges
    output_path = Path("/workspace/generated_training_challenges.json")
    generator.save_challenges(challenges, output_path)
    
    # Print statistics
    print("\nGeneration Statistics:")
    print(f"Total challenges: {len(challenges)}")
    
    size_counts = defaultdict(int)
    for challenge in challenges:
        # Extract size from challenge ID or calculate from first example
        if challenge.train:
            rows = len(challenge.train[0].input)
            cols = len(challenge.train[0].input[0])
            size_counts[f"{rows}x{cols}"] += 1
    
    print("\nChallenges by grid size:")
    for size in sorted(size_counts.keys()):
        print(f"  {size}: {size_counts[size]}")


if __name__ == "__main__":
    main()