import numpy as np

class QHGenOptimizer:
    def __init__(self, pop_size=20, mutation_rate=0.1):
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        # Default pool if no user constraints are given
        self.default_pool = ['Ni', 'Co', 'Fe', 'Cu', 'Mo', 'W', 'Pt', 'Pd', 'Ti']

    def initialize_population(self, allowed_elements=None):
        """
        Generates starting alloys. 
        If 'allowed_elements' is provided (e.g. ['Ni', 'Mo']), it ONLY uses those.
        """
        population = []
        
        # Decide which ingredients to use
        if allowed_elements:
            element_pool = allowed_elements
        else:
            element_pool = self.default_pool

        for _ in range(self.pop_size):
            # If user gave specific elements, try to use mostly those
            if allowed_elements:
                # Use all user elements, or a subset if the list is long
                num_to_pick = min(len(element_pool), 3) 
                chosen_els = np.random.choice(element_pool, num_to_pick, replace=False)
            else:
                # Random mode: Pick 2-3 random elements
                num_elements = np.random.randint(2, 4)
                chosen_els = np.random.choice(element_pool, num_elements, replace=False)
            
            # Assign random ratios that sum to 1.0 (The AI's job is to optimize this)
            ratios = np.random.dirichlet(np.ones(len(chosen_els)))
            
            comp = {el: float(r) for el, r in zip(chosen_els, ratios)}
            population.append(comp)
            
        return population

    def calculate_fitness(self, energy):
        """
        Target is 0 eV. Closer is better.
        Fitness = 1 / (Error + epsilon)
        """
        target = 0.0
        error = abs(energy - target)
        return 1.0 / (error + 1e-4)

    def evolve(self, population, fitness_scores):
        """
        Standard Genetic Algorithm: Select, Crossover, Mutate.
        """
        # (Simplified implementation for the prototype - keeps top 50% and mutates)
        # Sort by fitness
        sorted_pop = [x for _, x in sorted(zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True)]
        
        # Keep top 50% (Elitism)
        new_pop = sorted_pop[:self.pop_size // 2]
        
        # Fill rest with mutated versions of winners
        while len(new_pop) < self.pop_size:
            parent = new_pop[np.random.randint(0, len(new_pop))]
            child = self.mutate(parent)
            new_pop.append(child)
            
        return new_pop
    # ... inside QHGenOptimizer ...

    def calculate_fitness_with_exploration(self, energy, uncertainty, exploration_weight=0.1):
        """
        Acquisition Function: Lower Confidence Bound (LCB)
        We want to MINIMIZE energy.
        Score = Energy - (lambda * uncertainty)
        
        If Uncertainty is HIGH, the score gets lower (better), encouraging the AI 
        to check that area.
        """
        target = 0.0
        dist_from_zero = abs(energy - target)
        
        # We want to minimize distance from zero, but reward uncertainty
        # High fitness = Good candidate
        
        # Score: Lower is better
        acquisition_score = dist_from_zero - (exploration_weight * uncertainty)
        
        # Convert to fitness (Higher is better for GA selection)
        # Avoid division by zero
        return 1.0 / (acquisition_score + 1.0)

    def mutate(self, composition):
        """
        Slightly changes the ratios of an alloy.
        """
        new_comp = composition.copy()
        elements = list(new_comp.keys())
        
        # Pick an element to tweak
        target_el = np.random.choice(elements)
        change = np.random.normal(0, 0.05) # Shift by +/- 5%
        
        new_comp[target_el] = max(0.01, min(0.99, new_comp[target_el] + change))
        
        # Re-normalize to sum to 1.0
        total = sum(new_comp.values())
        for k in new_comp:
            new_comp[k] /= total
            
        return new_comp