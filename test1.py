# lsystem_art_regression.py
import random
import numpy as np
from PIL import Image, ImageDraw
from deap import base, creator, tools, gp, algorithms
import operator
import math
import os

# ---- L-System Renderer ----
def draw_lsystem(instructions, angle, length, img_size=(300, 300)):
    img = Image.new("RGB", img_size, "white")
    draw = ImageDraw.Draw(img)

    x, y = img_size[0] // 2, img_size[1] // 2
    stack = []
    theta = 0  # Initial direction

    for cmd in instructions:
        if cmd == "F":
            new_x = x + length * math.cos(math.radians(theta))
            new_y = y + length * math.sin(math.radians(theta))
            draw.line((x, y, new_x, new_y), fill="black")
            x, y = new_x, new_y
        elif cmd == "+":
            theta += angle
        elif cmd == "-":
            theta -= angle
        elif cmd == "[":
            stack.append((x, y, theta))
        elif cmd == "]":
            if stack:
                x, y, theta = stack.pop()

    return img

# ---- Target Image ----
target_img = Image.open("target/spiral.jpg").resize((300, 300)).convert("L")
target_arr = np.array(target_img)

# ---- Fitness Function ----
def image_fitness(individual):
    try:
        expr = individual[0]
        angle = float(individual[1])
        length = float(individual[2])
        rule = str(expr)
        if 'F' not in rule:
            return 1e6,
        instructions = apply_lsystem("F", rule, 5)
        img = draw_lsystem(instructions, angle=angle, length=length)
        cand_arr = np.array(img.convert("L"))
        mse = np.mean((target_arr - cand_arr) ** 2)
        return mse,
    except Exception:
        return 1e6,

# ---- L-System Expansion ----
def apply_lsystem(axiom, rule, iterations):
    for _ in range(iterations):
        axiom = "".join(rule if c == "F" else c for c in axiom)
    return axiom

# ---- Primitive Set ----
pset = gp.PrimitiveSet("MAIN", 0)
pset.addPrimitive(operator.concat, 2)
for _ in range(3): pset.addTerminal("F")
pset.addTerminal("+")
pset.addTerminal("-")
pset.addTerminal("[")
pset.addTerminal("]")

# ---- GP Setup ----
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genFull, pset=pset, min_=2, max_=4)

def generate_individual():
    expr = toolbox.expr()
    angle = random.uniform(5, 90)
    length = random.uniform(5, 20)
    return creator.Individual([gp.PrimitiveTree(expr), angle, length])

toolbox.register("individual", generate_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Crossover and mutation

def mate(ind1, ind2):
    gp.cxOnePoint(ind1[0], ind2[0])
    if random.random() < 0.5:
        ind1[1], ind2[1] = ind2[1], ind1[1]  # angle
    if random.random() < 0.5:
        ind1[2], ind2[2] = ind2[2], ind1[2]  # length
    return ind1, ind2

def mutate(ind):
    if random.random() < 0.7:
        ind[0] = toolbox.mutate(ind[0])[0]
    ind[1] = float(ind[1]) + random.uniform(-10, 10)
    ind[2] = float(ind[2]) + random.uniform(-2, 2)
    ind[1] = max(1, min(180, ind[1]))
    ind[2] = max(1, min(30, ind[2]))
    return ind,

toolbox.register("evaluate", image_fitness)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", mate)
toolbox.register("mutate", mutate)

# ---- Main Loop ----
def run_gp():
    if not os.path.exists("output"):
        os.makedirs("output")

    pop = toolbox.population(n=20)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.2, 15, stats=stats, halloffame=hof, verbose=True)

    best = hof[0]
    print("Best rule:", best[0])
    print(f"Angle: {best[1]:.2f}, Length: {best[2]:.2f}")

    best_instructions = apply_lsystem("F", str(best[0]), 5)
    best_img = draw_lsystem(best_instructions, angle=best[1], length=best[2])
    best_img.save("output/best.png")
    best_img.show()

if __name__ == "__main__":
    run_gp()
