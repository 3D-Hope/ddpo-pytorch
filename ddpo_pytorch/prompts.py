from importlib import resources
import os
import functools
import random
import inflect

IE = inflect.engine()
ASSETS_PATH = resources.files("ddpo_pytorch.assets")


@functools.cache
def _load_lines(path):
    """
    Load lines from a file. First tries to load from `path` directly, and if that doesn't exist, searches the
    `ddpo_pytorch/assets` directory for a file named `path`.
    """
    if not os.path.exists(path):
        newpath = ASSETS_PATH.joinpath(path)
    if not os.path.exists(newpath):
        raise FileNotFoundError(f"Could not find {path} or ddpo_pytorch.assets/{path}")
    path = newpath
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]


def from_file(path, low=None, high=None):
    prompts = _load_lines(path)[low:high]
    return random.choice(prompts), {}


def imagenet_all():
    return from_file("imagenet_classes.txt")


def imagenet_animals():
    return from_file("imagenet_classes.txt", 0, 398)


def imagenet_dogs():
    return from_file("imagenet_classes.txt", 151, 269)


def simple_animals():
    return from_file("simple_animals.txt")


def nouns_activities(nouns_file, activities_file):
    nouns = _load_lines(nouns_file)
    activities = _load_lines(activities_file)
    return f"{IE.a(random.choice(nouns))} {random.choice(activities)}", {}


def counting(nouns_file, low, high):
    nouns = _load_lines(nouns_file)
    number = IE.number_to_words(random.randint(low, high))
    noun = random.choice(nouns)
    plural_noun = IE.plural(noun)
    prompt = f"{number} {plural_noun}"
    metadata = {
        "questions": [
            f"How many {plural_noun} are there in this image?",
            f"What animal is in this image?",
        ],
        "answers": [
            number,
            noun,
        ],
    }
    return prompt, metadata


def manhattan_scenes():
    """
    Prompts for Manhattan-style urban scenes with strong geometric structure.
    These scenes typically have clear vanishing points and parallel lines.
    """
    scenes = [
        # Street views
        "a Manhattan street with tall buildings on both sides",
        "a New York City avenue with skyscrapers",
        "an urban street canyon with buildings",
        "a city street with perspective view",
        "a downtown street with high-rise buildings",
        
        # Building facades
        "a modern glass building facade",
        "a skyscraper with many windows",
        "an office building with grid pattern windows",
        "a tall building with repeating floors",
        "a contemporary building with geometric design",
        
        # Interior architectural
        "a long hallway with doors on both sides",
        "a modern corridor with perspective",
        "an office hallway with ceiling lights",
        "a hotel corridor with carpet pattern",
        "a subway station platform",
        
        # Bridges and infrastructure
        "a bridge with parallel cables",
        "a modern bridge with geometric structure",
        "a pedestrian bridge with railings",
        "an overpass with support beams",
        
        # Urban infrastructure
        "a parking garage with levels",
        "a stairwell with railings",
        "an escalator in a mall",
        "a modern train station interior",
        "an airport terminal hallway",
        
        # Architectural details
        "a building with fire escapes",
        "a facade with balconies",
        "a modern building with horizontal lines",
        "a structure with vertical columns",
        "a building with geometric patterns",
    ]
    
    return random.choice(scenes), {}


def architectural_interiors():
    """
    Interior architectural scenes with strong perspective and vanishing points.
    """
    interiors = [
        "a modern office interior with desks",
        "a library with bookshelves",
        "a museum gallery with paintings",
        "a shopping mall corridor",
        "a hotel lobby with columns",
        "a conference room with chairs",
        "a restaurant interior with tables",
        "a warehouse with storage racks",
        "a modern kitchen with cabinets",
        "a bathroom with tiles",
    ]
    
    return random.choice(interiors), {}


def geometric_structures():
    """
    Various geometric structures and patterns.
    """
    structures = [
        "a grid of windows on a building",
        "parallel lines converging to a point",
        "a checkerboard floor pattern",
        "a tiled wall with perspective",
        "a fence with vertical posts",
        "railroad tracks extending into distance",
        "a row of street lights",
        "a colonnade with pillars",
        "a pergola with parallel beams",
        "a lattice structure",
    ]
    
    return random.choice(structures), {}

