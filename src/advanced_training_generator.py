#!/usr/bin/env python3
"""
Advanced training pair generation system with sophisticated transformation rules.
Generates high-quality training pairs for ARC-AGI challenges.
"""

import json
import random
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum
import itertools
from collections import defaultdict
import math

import sys
sys.path.append('/workspace')

from src.models import Challenge, Example, GRID
from src.prompts.colors import color_map


class ComplexityLevel(Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


@dataclass
class AdvancedConfig:
    """Advanced configuration for training pair generation."""
    min_grid_size: int = 2
    max_grid_size: int = 24
    examples_per_challenge: int = 3
    total_challenges: int = 20000
    complexity_distribution: Dict[ComplexityLevel, float] = None
    transformation_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.complexity_distribution is None:
            self.complexity_distribution = {
                ComplexityLevel.SIMPLE: 0.4,
                ComplexityLevel.MEDIUM: 0.4,
                ComplexityLevel.COMPLEX: 0.2
            }
        
        if self.transformation_weights is None:
            self.transformation_weights = {
                "color_operations": 0.25,
                "shape_operations": 0.25,
                "pattern_recognition": 0.20,
                "mathematical": 0.15,
                "geometric": 0.15
            }


class AdvancedTrainingGenerator:
    """Advanced generator with sophisticated transformation rules."""
    
    def __init__(self, config: AdvancedConfig):
        self.config = config
        self.generated_challenges = []
        self.quality_metrics = defaultdict(list)
        
    def generate_random_grid(self, rows: int, cols: int, 
                           density: float = 0.3,
                           color_distribution: Optional[Dict[int, float]] = None) -> GRID:
        """Generate a random grid with controlled density and color distribution."""
        if color_distribution is None:
            # More realistic distribution: mostly black with some colored cells
            color_distribution = {
                0: 0.8,  # Black (background)
                1: 0.05, # Blue
                2: 0.05, # Red
                3: 0.03, # Green
                4: 0.02, # Yellow
                5: 0.02, # Grey
                6: 0.01, # Pink
                7: 0.01, # Orange
                8: 0.01, # Purple
                9: 0.00  # Brown (rare)
            }
        
        grid = []
        for i in range(rows):
            row = []
            for j in range(cols):
                if random.random() < density:
                    colors = list(color_distribution.keys())
                    weights = list(color_distribution.values())
                    color = np.random.choice(colors, p=weights)
                    row.append(color)
                else:
                    row.append(0)  # Black
            grid.append(row)
        
        return grid
    
    def generate_structured_grid(self, rows: int, cols: int, 
                               structure_type: str) -> GRID:
        """Generate a grid with specific structural patterns."""
        grid = [[0 for _ in range(cols)] for _ in range(rows)]
        
        if structure_type == "rectangles":
            # Add random rectangles
            num_rects = random.randint(1, min(3, rows * cols // 20))
            for _ in range(num_rects):
                color = random.randint(1, 9)
                x1, y1 = random.randint(0, rows-2), random.randint(0, cols-2)
                x2, y2 = random.randint(x1+1, rows-1), random.randint(y1+1, cols-1)
                for i in range(x1, x2+1):
                    for j in range(y1, y2+1):
                        grid[i][j] = color
        
        elif structure_type == "lines":
            # Add horizontal and vertical lines
            num_lines = random.randint(1, min(4, max(rows, cols) // 2))
            for _ in range(num_lines):
                color = random.randint(1, 9)
                if random.choice([True, False]):  # Horizontal line
                    row = random.randint(0, rows-1)
                    start_col = random.randint(0, cols-2)
                    end_col = random.randint(start_col+1, cols-1)
                    for j in range(start_col, end_col+1):
                        grid[row][j] = color
                else:  # Vertical line
                    col = random.randint(0, cols-1)
                    start_row = random.randint(0, rows-2)
                    end_row = random.randint(start_row+1, rows-1)
                    for i in range(start_row, end_row+1):
                        grid[i][col] = color
        
        elif structure_type == "diagonal":
            # Add diagonal patterns
            color = random.randint(1, 9)
            for i in range(rows):
                for j in range(cols):
                    if (i + j) % 2 == 0:
                        grid[i][j] = color
        
        elif structure_type == "spiral":
            # Create spiral pattern
            color = random.randint(1, 9)
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            x, y, d = 0, 0, 0
            visited = set()
            
            for _ in range(min(rows * cols, 15)):
                if 0 <= x < rows and 0 <= y < cols and (x, y) not in visited:
                    grid[x][y] = color
                    visited.add((x, y))
                
                nx, ny = x + directions[d][0], y + directions[d][1]
                if (nx < 0 or nx >= rows or ny < 0 or ny >= cols or (nx, ny) in visited):
                    d = (d + 1) % 4
                    nx, ny = x + directions[d][0], y + directions[d][1]
                
                x, y = nx, ny
        
        elif structure_type == "checkerboard":
            # Checkerboard pattern
            color1, color2 = random.randint(1, 9), random.randint(1, 9)
            while color2 == color1:
                color2 = random.randint(1, 9)
            
            for i in range(rows):
                for j in range(cols):
                    if (i + j) % 2 == 0:
                        grid[i][j] = color1
                    else:
                        grid[i][j] = color2
        
        return grid
    
    def generate_color_operation_challenge(self, grid_size: Tuple[int, int], 
                                         complexity: ComplexityLevel) -> Challenge:
        """Generate color-based transformation challenges."""
        rows, cols = grid_size
        examples = []
        
        # Choose transformation based on complexity
        if complexity == ComplexityLevel.SIMPLE:
            transformations = ["replace_color", "invert_color", "shift_color"]
        elif complexity == ComplexityLevel.MEDIUM:
            transformations = ["conditional_replace", "color_gradient", "color_based_pattern"]
        else:
            transformations = ["multi_color_transform", "color_sequence", "complex_conditional"]
        
        rule_type = random.choice(transformations)
        
        for i in range(self.config.examples_per_challenge):
            input_grid = self.generate_random_grid(rows, cols)
            output_grid = self.apply_color_operation(input_grid, rule_type, complexity)
            
            examples.append(Example(
                input=input_grid,
                output=output_grid
            ))
        
        test_input = self.generate_random_grid(rows, cols)
        test_output = self.apply_color_operation(test_input, rule_type, complexity)
        
        challenge_id = f"color_{rows}x{cols}_{complexity.value}_{random.randint(1000, 9999)}"
        return Challenge(
            id=challenge_id,
            train=examples,
            test=[Example(input=test_input, output=test_output)]
        )
    
    def apply_color_operation(self, grid: GRID, rule_type: str, 
                            complexity: ComplexityLevel) -> GRID:
        """Apply color-based transformation."""
        grid = [row[:] for row in grid]  # Deep copy
        rows, cols = len(grid), len(grid[0])
        
        if rule_type == "replace_color":
            # Simple color replacement
            from_color = random.randint(1, 9)
            to_color = random.randint(1, 9)
            while to_color == from_color:
                to_color = random.randint(1, 9)
            
            for i in range(rows):
                for j in range(cols):
                    if grid[i][j] == from_color:
                        grid[i][j] = to_color
        
        elif rule_type == "invert_color":
            # Invert non-black colors
            for i in range(rows):
                for j in range(cols):
                    if grid[i][j] != 0:
                        grid[i][j] = 10 - grid[i][j]
        
        elif rule_type == "shift_color":
            # Shift all colors by a fixed amount
            shift = random.randint(1, 8)
            for i in range(rows):
                for j in range(cols):
                    if grid[i][j] != 0:
                        grid[i][j] = (grid[i][j] + shift) % 10
        
        elif rule_type == "conditional_replace":
            # Replace color based on position or neighbors
            for i in range(rows):
                for j in range(cols):
                    if grid[i][j] != 0:
                        # Replace based on position
                        if (i + j) % 2 == 0:
                            grid[i][j] = 2  # Red for even positions
                        else:
                            grid[i][j] = 1  # Blue for odd positions
        
        elif rule_type == "color_gradient":
            # Create gradient based on distance from center
            center_x, center_y = rows // 2, cols // 2
            max_dist = max(center_x, center_y, rows - center_x, cols - center_y)
            
            for i in range(rows):
                for j in range(cols):
                    if grid[i][j] != 0:
                        dist = ((i - center_x) ** 2 + (j - center_y) ** 2) ** 0.5
                        normalized_dist = min(dist / max_dist, 1.0)
                        color_value = int(normalized_dist * 9) + 1
                        grid[i][j] = color_value
        
        elif rule_type == "multi_color_transform":
            # Complex multi-color transformation
            color_map = {i: (i + 1) % 10 for i in range(1, 10)}
            color_map[0] = 0
            
            for i in range(rows):
                for j in range(cols):
                    grid[i][j] = color_map[grid[i][j]]
        
        return grid
    
    def generate_shape_operation_challenge(self, grid_size: Tuple[int, int], 
                                         complexity: ComplexityLevel) -> Challenge:
        """Generate shape-based transformation challenges."""
        rows, cols = grid_size
        examples = []
        
        if complexity == ComplexityLevel.SIMPLE:
            transformations = ["rotate_90", "flip_horizontal", "flip_vertical"]
        elif complexity == ComplexityLevel.MEDIUM:
            transformations = ["scale_2x", "move_shapes", "extract_shapes"]
        else:
            transformations = ["complex_rotation", "shape_morphing", "multi_shape_op"]
        
        rule_type = random.choice(transformations)
        
        for i in range(self.config.examples_per_challenge):
            input_grid = self.generate_structured_grid(rows, cols, "rectangles")
            output_grid = self.apply_shape_operation(input_grid, rule_type, complexity)
            
            examples.append(Example(
                input=input_grid,
                output=output_grid
            ))
        
        test_input = self.generate_structured_grid(rows, cols, "rectangles")
        test_output = self.apply_shape_operation(test_input, rule_type, complexity)
        
        challenge_id = f"shape_{rows}x{cols}_{complexity.value}_{random.randint(1000, 9999)}"
        return Challenge(
            id=challenge_id,
            train=examples,
            test=[Example(input=test_input, output=test_output)]
        )
    
    def apply_shape_operation(self, grid: GRID, rule_type: str, 
                            complexity: ComplexityLevel) -> GRID:
        """Apply shape-based transformation."""
        grid = [row[:] for row in grid]  # Deep copy
        rows, cols = len(grid), len(grid[0])
        
        if rule_type == "rotate_90":
            # Rotate 90 degrees clockwise
            return [[grid[rows-1-j][i] for j in range(rows)] for i in range(cols)]
        
        elif rule_type == "flip_horizontal":
            # Flip horizontally
            return [row[::-1] for row in grid]
        
        elif rule_type == "flip_vertical":
            # Flip vertically
            return grid[::-1]
        
        elif rule_type == "scale_2x":
            # Scale up by 2x (if possible)
            if rows * 2 <= 24 and cols * 2 <= 24:
                new_grid = [[0 for _ in range(cols * 2)] for _ in range(rows * 2)]
                for i in range(rows):
                    for j in range(cols):
                        if grid[i][j] != 0:
                            new_grid[i*2][j*2] = grid[i][j]
                            if i*2+1 < rows*2 and j*2+1 < cols*2:
                                new_grid[i*2+1][j*2] = grid[i][j]
                                new_grid[i*2][j*2+1] = grid[i][j]
                                new_grid[i*2+1][j*2+1] = grid[i][j]
                return new_grid
            else:
                return grid
        
        elif rule_type == "move_shapes":
            # Move all non-zero elements by offset
            dx, dy = random.randint(-1, 1), random.randint(-1, 1)
            new_grid = [[0 for _ in range(cols)] for _ in range(rows)]
            for i in range(rows):
                for j in range(cols):
                    if grid[i][j] != 0:
                        new_i, new_j = i + dx, j + dy
                        if 0 <= new_i < rows and 0 <= new_j < cols:
                            new_grid[new_i][new_j] = grid[i][j]
            return new_grid
        
        return grid
    
    def generate_pattern_recognition_challenge(self, grid_size: Tuple[int, int], 
                                             complexity: ComplexityLevel) -> Challenge:
        """Generate pattern recognition challenges."""
        rows, cols = grid_size
        examples = []
        
        if complexity == ComplexityLevel.SIMPLE:
            transformations = ["find_edges", "count_objects", "detect_symmetry"]
        elif complexity == ComplexityLevel.MEDIUM:
            transformations = ["pattern_matching", "feature_extraction", "contour_detection"]
        else:
            transformations = ["complex_pattern", "multi_pattern", "hierarchical_pattern"]
        
        rule_type = random.choice(transformations)
        
        for i in range(self.config.examples_per_challenge):
            input_grid = self.generate_structured_grid(rows, cols, "lines")
            output_grid = self.apply_pattern_recognition(input_grid, rule_type, complexity)
            
            examples.append(Example(
                input=input_grid,
                output=output_grid
            ))
        
        test_input = self.generate_structured_grid(rows, cols, "lines")
        test_output = self.apply_pattern_recognition(test_input, rule_type, complexity)
        
        challenge_id = f"pattern_{rows}x{cols}_{complexity.value}_{random.randint(1000, 9999)}"
        return Challenge(
            id=challenge_id,
            train=examples,
            test=[Example(input=test_input, output=test_output)]
        )
    
    def apply_pattern_recognition(self, grid: GRID, rule_type: str, 
                                complexity: ComplexityLevel) -> GRID:
        """Apply pattern recognition transformation."""
        grid = [row[:] for row in grid]  # Deep copy
        rows, cols = len(grid), len(grid[0])
        new_grid = [[0 for _ in range(cols)] for _ in range(rows)]
        
        if rule_type == "find_edges":
            # Simple edge detection
            for i in range(rows):
                for j in range(cols):
                    if grid[i][j] != 0:
                        # Check 4-connected neighbors
                        neighbors = []
                        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < rows and 0 <= nj < cols:
                                neighbors.append(grid[ni][nj])
                        
                        # If any neighbor is different or black, it's an edge
                        if any(n != grid[i][j] for n in neighbors):
                            new_grid[i][j] = 2  # Red for edges
                        else:
                            new_grid[i][j] = 1  # Blue for interior
        
        elif rule_type == "count_objects":
            # Count connected components
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
            for i in range(rows):
                for j in range(cols):
                    if grid[i][j] != 0:
                        sym_j = cols - 1 - j
                        if sym_j >= 0 and grid[i][sym_j] == grid[i][j]:
                            new_grid[i][j] = 2  # Red for symmetric
                        else:
                            new_grid[i][j] = 1  # Blue for asymmetric
        
        return new_grid
    
    def generate_mathematical_challenge(self, grid_size: Tuple[int, int], 
                                      complexity: ComplexityLevel) -> Challenge:
        """Generate mathematical transformation challenges."""
        rows, cols = grid_size
        examples = []
        
        if complexity == ComplexityLevel.SIMPLE:
            transformations = ["sum_neighbors", "multiply_position", "modulo_op"]
        elif complexity == ComplexityLevel.MEDIUM:
            transformations = ["gradient", "distance_transform", "convolution"]
        else:
            transformations = ["complex_math", "multi_operation", "recursive_math"]
        
        rule_type = random.choice(transformations)
        
        for i in range(self.config.examples_per_challenge):
            input_grid = self.generate_math_grid(rows, cols)
            output_grid = self.apply_mathematical_operation(input_grid, rule_type, complexity)
            
            examples.append(Example(
                input=input_grid,
                output=output_grid
            ))
        
        test_input = self.generate_math_grid(rows, cols)
        test_output = self.apply_mathematical_operation(test_input, rule_type, complexity)
        
        challenge_id = f"math_{rows}x{cols}_{complexity.value}_{random.randint(1000, 9999)}"
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
                # Generate numbers suitable for math operations
                value = random.randint(0, 4)  # Smaller range for better math operations
                row.append(value)
            grid.append(row)
        return grid
    
    def apply_mathematical_operation(self, grid: GRID, rule_type: str, 
                                   complexity: ComplexityLevel) -> GRID:
        """Apply mathematical transformation."""
        grid = [row[:] for row in grid]  # Deep copy
        rows, cols = len(grid), len(grid[0])
        new_grid = [[0 for _ in range(cols)] for _ in range(rows)]
        
        if rule_type == "sum_neighbors":
            # Sum of 8-connected neighbors
            for i in range(rows):
                for j in range(cols):
                    total = 0
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < rows and 0 <= nj < cols and (di != 0 or dj != 0):
                                total += grid[ni][nj]
                    new_grid[i][j] = min(total % 10, 9)
        
        elif rule_type == "multiply_position":
            # Multiply by position
            for i in range(rows):
                for j in range(cols):
                    new_grid[i][j] = (grid[i][j] * (i + j + 1)) % 10
        
        elif rule_type == "modulo_op":
            # Modulo operation
            mod_value = random.randint(2, 5)
            for i in range(rows):
                for j in range(cols):
                    new_grid[i][j] = grid[i][j] % mod_value
        
        return new_grid
    
    def generate_geometric_challenge(self, grid_size: Tuple[int, int], 
                                   complexity: ComplexityLevel) -> Challenge:
        """Generate geometric transformation challenges."""
        rows, cols = grid_size
        examples = []
        
        if complexity == ComplexityLevel.SIMPLE:
            transformations = ["distance_center", "angle_calculation", "coordinate_transform"]
        elif complexity == ComplexityLevel.MEDIUM:
            transformations = ["voronoi", "delaunay", "convex_hull"]
        else:
            transformations = ["complex_geometric", "multi_geometric", "hierarchical_geometric"]
        
        rule_type = random.choice(transformations)
        
        for i in range(self.config.examples_per_challenge):
            input_grid = self.generate_geometric_grid(rows, cols)
            output_grid = self.apply_geometric_operation(input_grid, rule_type, complexity)
            
            examples.append(Example(
                input=input_grid,
                output=output_grid
            ))
        
        test_input = self.generate_geometric_grid(rows, cols)
        test_output = self.apply_geometric_operation(test_input, rule_type, complexity)
        
        challenge_id = f"geo_{rows}x{cols}_{complexity.value}_{random.randint(1000, 9999)}"
        return Challenge(
            id=challenge_id,
            train=examples,
            test=[Example(input=test_input, output=test_output)]
        )
    
    def generate_geometric_grid(self, rows: int, cols: int) -> GRID:
        """Generate a grid with geometric features."""
        grid = [[0 for _ in range(cols)] for _ in range(rows)]
        
        # Add geometric points
        num_points = random.randint(2, min(5, rows * cols // 8))
        for _ in range(num_points):
            x, y = random.randint(0, rows-1), random.randint(0, cols-1)
            color = random.randint(1, 9)
            grid[x][y] = color
        
        return grid
    
    def apply_geometric_operation(self, grid: GRID, rule_type: str, 
                                complexity: ComplexityLevel) -> GRID:
        """Apply geometric transformation."""
        grid = [row[:] for row in grid]  # Deep copy
        rows, cols = len(grid), len(grid[0])
        new_grid = [[0 for _ in range(cols)] for _ in range(rows)]
        
        if rule_type == "distance_center":
            # Distance from center
            center_x, center_y = rows // 2, cols // 2
            max_dist = max(center_x, center_y, rows - center_x, cols - center_y)
            
            for i in range(rows):
                for j in range(cols):
                    if grid[i][j] != 0:
                        dist = ((i - center_x) ** 2 + (j - center_y) ** 2) ** 0.5
                        normalized_dist = min(dist / max_dist, 1.0)
                        color_value = int(normalized_dist * 9) + 1
                        new_grid[i][j] = color_value
        
        elif rule_type == "angle_calculation":
            # Calculate angle from center
            center_x, center_y = rows // 2, cols // 2
            
            for i in range(rows):
                for j in range(cols):
                    if grid[i][j] != 0:
                        dx, dy = i - center_x, j - center_y
                        if dx == 0 and dy == 0:
                            angle = 0
                        else:
                            angle = math.atan2(dy, dx)
                            angle = (angle + math.pi) / (2 * math.pi)  # Normalize to [0, 1]
                        color_value = int(angle * 9) + 1
                        new_grid[i][j] = color_value
        
        return new_grid
    
    def generate_challenge(self, grid_size: Tuple[int, int]) -> Challenge:
        """Generate a random challenge of the specified grid size."""
        # Choose complexity level
        complexity_choice = random.random()
        cumulative = 0
        complexity = ComplexityLevel.SIMPLE
        
        for comp_level, prob in self.config.complexity_distribution.items():
            cumulative += prob
            if complexity_choice <= cumulative:
                complexity = comp_level
                break
        
        # Choose transformation type
        transformation_choice = random.random()
        cumulative = 0
        
        if transformation_choice <= self.config.transformation_weights["color_operations"]:
            return self.generate_color_operation_challenge(grid_size, complexity)
        elif transformation_choice <= (self.config.transformation_weights["color_operations"] + 
                                     self.config.transformation_weights["shape_operations"]):
            return self.generate_shape_operation_challenge(grid_size, complexity)
        elif transformation_choice <= (self.config.transformation_weights["color_operations"] + 
                                     self.config.transformation_weights["shape_operations"] +
                                     self.config.transformation_weights["pattern_recognition"]):
            return self.generate_pattern_recognition_challenge(grid_size, complexity)
        elif transformation_choice <= (self.config.transformation_weights["color_operations"] + 
                                     self.config.transformation_weights["shape_operations"] +
                                     self.config.transformation_weights["pattern_recognition"] +
                                     self.config.transformation_weights["mathematical"]):
            return self.generate_mathematical_challenge(grid_size, complexity)
        else:
            return self.generate_geometric_challenge(grid_size, complexity)
    
    def validate_challenge_quality(self, challenge: Challenge) -> bool:
        """Validate the quality of a generated challenge."""
        # Check that all examples have the same input/output dimensions
        for example in challenge.train + challenge.test:
            if len(example.input) != len(example.output):
                return False
            if len(example.input[0]) != len(example.output[0]):
                return False
        
        # Check that transformations are not trivial (all same color)
        for example in challenge.train:
            input_colors = set()
            output_colors = set()
            for row in example.input:
                input_colors.update(row)
            for row in example.output:
                output_colors.update(row)
            
            # Should have some variation
            if len(input_colors) < 2 or len(output_colors) < 2:
                return False
        
        return True
    
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
            
            generated_for_size = 0
            attempts = 0
            max_attempts = challenges_per_size * 2  # Allow some failures
            
            while generated_for_size < challenges_per_size and attempts < max_attempts:
                try:
                    challenge = self.generate_challenge((size, size))
                    
                    if self.validate_challenge_quality(challenge):
                        challenges.append(challenge)
                        generated_for_size += 1
                        
                        if generated_for_size % 50 == 0:
                            print(f"  Generated {generated_for_size}/{challenges_per_size} challenges for {size}x{size}")
                    
                    attempts += 1
                
                except Exception as e:
                    print(f"  Error generating challenge for {size}x{size}: {e}")
                    attempts += 1
                    continue
            
            print(f"  Completed {generated_for_size}/{challenges_per_size} challenges for {size}x{size}")
        
        # Add remaining challenges to reach total
        remaining = self.config.total_challenges - len(challenges)
        if remaining > 0:
            print(f"Adding {remaining} additional challenges...")
            attempts = 0
            max_attempts = remaining * 2
            
            while len(challenges) < self.config.total_challenges and attempts < max_attempts:
                try:
                    size = random.randint(self.config.min_grid_size, self.config.max_grid_size)
                    challenge = self.generate_challenge((size, size))
                    
                    if self.validate_challenge_quality(challenge):
                        challenges.append(challenge)
                    
                    attempts += 1
                except Exception as e:
                    print(f"Error generating additional challenge: {e}")
                    attempts += 1
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
    """Main function to generate advanced training pairs."""
    config = AdvancedConfig(
        min_grid_size=2,
        max_grid_size=24,
        examples_per_challenge=3,
        total_challenges=20000,
        complexity_distribution={
            ComplexityLevel.SIMPLE: 0.4,
            ComplexityLevel.MEDIUM: 0.4,
            ComplexityLevel.COMPLEX: 0.2
        },
        transformation_weights={
            "color_operations": 0.25,
            "shape_operations": 0.25,
            "pattern_recognition": 0.20,
            "mathematical": 0.15,
            "geometric": 0.15
        }
    )
    
    generator = AdvancedTrainingGenerator(config)
    challenges = generator.generate_all_challenges()
    
    # Save challenges
    output_path = Path("/workspace/advanced_training_challenges.json")
    generator.save_challenges(challenges, output_path)
    
    # Print statistics
    print("\nGeneration Statistics:")
    print(f"Total challenges: {len(challenges)}")
    
    size_counts = defaultdict(int)
    complexity_counts = defaultdict(int)
    type_counts = defaultdict(int)
    
    for challenge in challenges:
        # Extract size from challenge ID or calculate from first example
        if challenge.train:
            rows = len(challenge.train[0].input)
            cols = len(challenge.train[0].input[0])
            size_counts[f"{rows}x{cols}"] += 1
        
        # Extract complexity and type from challenge ID
        parts = challenge.id.split('_')
        if len(parts) >= 3:
            type_counts[parts[0]] += 1
            if len(parts) >= 4:
                complexity_counts[parts[2]] += 1
    
    print("\nChallenges by grid size:")
    for size in sorted(size_counts.keys()):
        print(f"  {size}: {size_counts[size]}")
    
    print("\nChallenges by type:")
    for type_name in sorted(type_counts.keys()):
        print(f"  {type_name}: {type_counts[type_name]}")
    
    print("\nChallenges by complexity:")
    for complexity in sorted(complexity_counts.keys()):
        print(f"  {complexity}: {complexity_counts[complexity]}")


if __name__ == "__main__":
    main()