import random
import numpy as np
from typing import List, Dict, Tuple, Set
import copy
import time
import scipy.special

# Hashim Abdulla
# Program 2: Genetic Algorithm for Course Scheduling
# CS 461 - Artificial Intelligence

# Define constants
ACTIVITIES = ["SLA100A", "SLA100B", "SLA191A", "SLA191B", "SLA201", "SLA291", "SLA303", "SLA304", "SLA394", "SLA449", "SLA451"]
ROOMS = ["Slater 003", "Roman 216", "Loft 206", "Roman 201", "Loft 310", "Beach 201", "Beach 301", "Logos 325", "Frank 119"]
TIMES = ["10 AM", "11 AM", "12 PM", "1 PM", "2 PM", "3 PM"]
FACILITATORS = ["Lock", "Glen", "Banks", "Richards", "Shaw", "Singer", "Uther", "Tyler", "Numen", "Zeldin"]

# Room capacities
ROOM_CAPACITY = {
    "Slater 003": 45,
    "Roman 216": 30,
    "Loft 206": 75,
    "Roman 201": 50,
    "Loft 310": 108,
    "Beach 201": 60,
    "Beach 301": 75,
    "Logos 325": 450,
    "Frank 119": 60
}

# Expected enrollment for each activity
EXPECTED_ENROLLMENT = {
    "SLA100A": 50, "SLA100B": 50,
    "SLA191A": 50, "SLA191B": 50,
    "SLA201": 50,
    "SLA291": 50,
    "SLA303": 60,
    "SLA304": 25,
    "SLA394": 20,
    "SLA449": 60,
    "SLA451": 100
}

# Preferred facilitators for each activity
PREFERRED_FACILITATORS = {
    "SLA100A": ["Glen", "Lock", "Banks", "Zeldin"],
    "SLA100B": ["Glen", "Lock", "Banks", "Zeldin"],
    "SLA191A": ["Glen", "Lock", "Banks", "Zeldin"],
    "SLA191B": ["Glen", "Lock", "Banks", "Zeldin"],
    "SLA201": ["Glen", "Banks", "Zeldin", "Shaw"],
    "SLA291": ["Lock", "Banks", "Zeldin", "Singer"],
    "SLA303": ["Glen", "Zeldin", "Banks"],
    "SLA304": ["Glen", "Banks", "Tyler"],
    "SLA394": ["Tyler", "Singer"],
    "SLA449": ["Tyler", "Singer", "Shaw"],
    "SLA451": ["Tyler", "Singer", "Shaw"]
}

# Other facilitators for each activity
OTHER_FACILITATORS = {
    "SLA100A": ["Numen", "Richards"],
    "SLA100B": ["Numen", "Richards"],
    "SLA191A": ["Numen", "Richards"],
    "SLA191B": ["Numen", "Richards"],
    "SLA201": ["Numen", "Richards", "Singer"],
    "SLA291": ["Numen", "Richards", "Shaw", "Tyler"],
    "SLA303": ["Numen", "Singer", "Shaw"],
    "SLA304": ["Numen", "Singer", "Shaw", "Richards", "Uther", "Zeldin"],
    "SLA394": ["Richards", "Zeldin"],
    "SLA449": ["Zeldin", "Uther"],
    "SLA451": ["Zeldin", "Uther", "Richards", "Banks"]
}

# Helper function to convert time slot to hours
def time_to_hours(time_slot):
    """Convert time slot to numerical hours for easier comparison."""
    if time_slot == "10 AM":
        return 10
    elif time_slot == "11 AM":
        return 11
    elif time_slot == "12 PM":
        return 12
    elif time_slot == "1 PM":
        return 13
    elif time_slot == "2 PM":
        return 14
    elif time_slot == "3 PM":
        return 15
    return -1  # Invalid time

# Define a class to represent a schedule
class Schedule:
    def __init__(self):
        # Initialize with random assignments
        self.assignments = {}
        for activity in ACTIVITIES:
            room = random.choice(ROOMS)
            time = random.choice(TIMES)
            facilitator = random.choice(FACILITATORS)
            self.assignments[activity] = {
                "room": room,
                "time": time,
                "facilitator": facilitator
            }
        self.fitness = None

    def calculate_fitness(self) -> float:
        """Calculate the fitness of this schedule based on the given criteria."""
        total_fitness = 0

        # Check for room conflicts (same room at same time)
        time_room_pairs = {}
        facilitator_time_count = {}
        facilitator_total_count = {}
        facilitator_consecutive_times = {}

        for activity, details in self.assignments.items():
            room = details["room"]
            time = details["time"]
            facilitator = details["facilitator"]

            # Initialize counts
            if facilitator not in facilitator_total_count:
                facilitator_total_count[facilitator] = 0
            facilitator_total_count[facilitator] += 1

            if facilitator not in facilitator_consecutive_times:
                facilitator_consecutive_times[facilitator] = []
            facilitator_consecutive_times[facilitator].append(time_to_hours(time))

            time_room_key = f"{time}_{room}"
            if time_room_key not in time_room_pairs:
                time_room_pairs[time_room_key] = []
            time_room_pairs[time_room_key].append(activity)

            facilitator_time_key = f"{facilitator}_{time}"
            if facilitator_time_key not in facilitator_time_count:
                facilitator_time_count[facilitator_time_key] = 0
            facilitator_time_count[facilitator_time_key] += 1

        # Process each activity's fitness
        for activity, details in self.assignments.items():
            activity_fitness = 0
            room = details["room"]
            time = details["time"]
            facilitator = details["facilitator"]

            # Check for room conflicts
            time_room_key = f"{time}_{room}"
            if len(time_room_pairs[time_room_key]) > 1:
                activity_fitness -= 0.5

            # Check room size
            capacity = ROOM_CAPACITY[room]
            enrollment = EXPECTED_ENROLLMENT[activity]

            if capacity < enrollment:
                activity_fitness -= 0.5
            elif capacity > 6 * enrollment:
                activity_fitness -= 0.4
            elif capacity > 3 * enrollment:
                activity_fitness -= 0.2
            else:
                activity_fitness += 0.3

            # Check facilitator preference
            if facilitator in PREFERRED_FACILITATORS[activity]:
                activity_fitness += 0.5
            elif facilitator in OTHER_FACILITATORS[activity]:
                activity_fitness += 0.2
            else:
                activity_fitness -= 0.1

            # Check facilitator load for this time slot
            facilitator_time_key = f"{facilitator}_{time}"
            if facilitator_time_count[facilitator_time_key] == 1:
                activity_fitness += 0.2
            elif facilitator_time_count[facilitator_time_key] > 1:
                activity_fitness -= 0.2

            # Check total facilitator load
            if facilitator_total_count[facilitator] > 4:
                activity_fitness -= 0.5
            elif facilitator_total_count[facilitator] < 3 and facilitator != "Tyler":
                activity_fitness -= 0.4

            # Activity-specific adjustments
            # SLA100A and SLA100B time separation
            if activity == "SLA100A" or activity == "SLA100B":
                time_A = time_to_hours(self.assignments["SLA100A"]["time"])
                time_B = time_to_hours(self.assignments["SLA100B"]["time"])
                if abs(time_A - time_B) > 4:  # More than 4 hours apart
                    activity_fitness += 0.5
                elif time_A == time_B:  # Same time slot
                    activity_fitness -= 0.5

            # SLA191A and SLA191B time separation
            if activity == "SLA191A" or activity == "SLA191B":
                time_A = time_to_hours(self.assignments["SLA191A"]["time"])
                time_B = time_to_hours(self.assignments["SLA191B"]["time"])
                if abs(time_A - time_B) > 4:  # More than 4 hours apart
                    activity_fitness += 0.5
                elif time_A == time_B:  # Same time slot
                    activity_fitness -= 0.5

            # SLA101 and SLA191 consecutive time slots
            if activity in ["SLA100A", "SLA100B", "SLA191A", "SLA191B"]:
                for sla100 in ["SLA100A", "SLA100B"]:
                    for sla191 in ["SLA191A", "SLA191B"]:
                        time_100 = time_to_hours(self.assignments[sla100]["time"])
                        time_191 = time_to_hours(self.assignments[sla191]["time"])

                        # Check if consecutive (1 hour apart)
                        if abs(time_100 - time_191) == 1:
                            activity_fitness += 0.5

                            # Check if one is in Roman or Beach and the other isn't
                            room_100 = self.assignments[sla100]["room"]
                            room_191 = self.assignments[sla191]["room"]

                            if (("Roman" in room_100 or "Beach" in room_100) and 
                                not ("Roman" in room_191 or "Beach" in room_191)) or \
                               (("Roman" in room_191 or "Beach" in room_191) and 
                                not ("Roman" in room_100 or "Beach" in room_100)):
                                activity_fitness -= 0.4

                        # Check if separated by 1 hour (2 hours apart)
                        elif abs(time_100 - time_191) == 2:
                            activity_fitness += 0.25

                        # Check if in same time slot
                        elif time_100 == time_191:
                            activity_fitness -= 0.25

            total_fitness += activity_fitness

        # Check for consecutive facilitator time slots
        for facilitator, times in facilitator_consecutive_times.items():
            times.sort()
            for i in range(len(times) - 1):
                if times[i+1] - times[i] == 1:  # Consecutive times
                    # Check if this involves SLA101 and SLA191 sections
                    # This is already accounted for in the activity-specific section
                    pass

        self.fitness = total_fitness
        return total_fitness

    def mutate(self, mutation_rate: float) -> None:
        """Apply mutation to the schedule with the given mutation rate."""
        for activity in self.assignments:
            if random.random() < mutation_rate:
                # Choose what to mutate: room, time, or facilitator
                attribute = random.choice(["room", "time", "facilitator"])
                if attribute == "room":
                    self.assignments[activity]["room"] = random.choice(ROOMS)
                elif attribute == "time":
                    self.assignments[activity]["time"] = random.choice(TIMES)
                else:  # attribute == "facilitator"
                    self.assignments[activity]["facilitator"] = random.choice(FACILITATORS)


def create_initial_population(population_size: int) -> List[Schedule]:
    """Create an initial population of random schedules."""
    return [Schedule() for _ in range(population_size)]


def softmax_selection(population: List[Schedule]) -> Tuple[Schedule, Schedule]:
    """Select two parents from the population using softmax-based selection."""
    # Get all fitness scores
    fitness_scores = np.array([schedule.fitness for schedule in population])

    # Apply softmax to convert fitness scores to probabilities
    probabilities = scipy.special.softmax(fitness_scores)

    # Select two parents based on probabilities
    parent_indices = np.random.choice(len(population), size=2, replace=False, p=probabilities)
    parent1 = population[parent_indices[0]]
    parent2 = population[parent_indices[1]]

    return parent1, parent2


def crossover(parent1: Schedule, parent2: Schedule) -> Schedule:
    """Create a child by crossing over two parents."""
    child = Schedule()

    # For each activity, randomly choose which parent to inherit from
    for activity in ACTIVITIES:
        if random.random() < 0.5:
            child.assignments[activity] = copy.deepcopy(parent1.assignments[activity])
        else:
            child.assignments[activity] = copy.deepcopy(parent2.assignments[activity])

    return child


def genetic_algorithm(population_size: int, num_generations: int, mutation_rate: float):
    """Run the genetic algorithm."""
    # Create initial population
    population = create_initial_population(population_size)

    # Calculate initial fitness
    for schedule in population:
        schedule.calculate_fitness()

    # Sort population by fitness (descending)
    population.sort(key=lambda x: x.fitness, reverse=True)

    # Track best fitness over generations
    best_fitness_history = []
    avg_fitness_history = []

    # Main GA loop
    for generation in range(num_generations):
        # Create new population
        new_population = []

        # Elitism: keep the best individual
        new_population.append(copy.deepcopy(population[0]))

        # Create the rest of the new population
        while len(new_population) < population_size:
            # Select parents using softmax selection
            parent1, parent2 = softmax_selection(population)

            # Create child
            child = crossover(parent1, parent2)

            # Apply mutation
            child.mutate(mutation_rate)

            # Calculate fitness
            child.calculate_fitness()

            # Add to new population
            new_population.append(child)

        # Replace old population
        population = new_population

        # Sort by fitness
        population.sort(key=lambda x: x.fitness, reverse=True)

        # Track best and average fitness
        best_fitness = population[0].fitness
        avg_fitness = sum(p.fitness for p in population) / len(population)
        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(avg_fitness)

        # Print progress
        if generation % 10 == 0:
            print(f"Generation {generation}: Best fitness = {best_fitness:.2f}, Avg fitness = {avg_fitness:.2f}")

        # Check convergence after 100 generations
        if generation >= 100:
            improvement = (avg_fitness - avg_fitness_history[generation - 100]) / abs(avg_fitness_history[generation - 100])
            if improvement < 0.01:  # Less than 1% improvement
                print(f"Converged after {generation} generations")
                break

    return population[0], best_fitness_history, avg_fitness_history


def print_schedule(schedule: Schedule, filename: str):
    """Print the schedule to a file."""
    with open(filename, 'w') as f:
        f.write("Final Schedule\n")
        f.write("=============\n\n")
        f.write(f"Fitness: {schedule.fitness:.2f}\n\n")

        # Print by time slot
        for time in TIMES:
            f.write(f"Time: {time}\n")
            f.write("---------------------\n")

            activities_at_time = []
            for activity, details in schedule.assignments.items():
                if details["time"] == time:
                    activities_at_time.append((activity, details["room"], details["facilitator"]))

            activities_at_time.sort()  # Sort by activity name

            for activity, room, facilitator in activities_at_time:
                enrollment = EXPECTED_ENROLLMENT[activity]
                capacity = ROOM_CAPACITY[room]
                f.write(f"{activity} - Room: {room} (Capacity: {capacity}, Enrollment: {enrollment}), Facilitator: {facilitator}\n")

            f.write("\n")

        # Print by facilitator
        f.write("Facilitator Assignments\n")
        f.write("======================\n\n")

        for facilitator in FACILITATORS:
            f.write(f"Facilitator: {facilitator}\n")
            f.write("---------------------\n")

            activities_by_facilitator = []
            for activity, details in schedule.assignments.items():
                if details["facilitator"] == facilitator:
                    activities_by_facilitator.append((activity, details["time"], details["room"]))

            # Sort by time
            activities_by_facilitator.sort(key=lambda x: TIMES.index(x[1]))

            for activity, time, room in activities_by_facilitator:
                f.write(f"{activity} - Time: {time}, Room: {room}\n")

            f.write(f"Total activities: {len(activities_by_facilitator)}\n\n")


def main():
    # Set parameters
    population_size = 500  # Meets requirement N >= 500
    num_generations = 500
    initial_mutation_rate = 0.01  # Initial mutation rate

    start_time = time.time()

    # Run genetic algorithm with initial mutation rate
    best_schedule, best_fitness_history, avg_fitness_history = genetic_algorithm(
        population_size, num_generations, initial_mutation_rate
    )

    initial_best_fitness = best_schedule.fitness
    print(f"\nInitial best fitness with mutation rate {initial_mutation_rate}: {initial_best_fitness:.2f}")

    # Halve mutation rate until results stabilize
    current_mutation_rate = initial_mutation_rate / 2
    previous_best_fitness = initial_best_fitness
    improved = True

    while improved and current_mutation_rate >= 0.0001:  # Lower bound to prevent infinite loop
        print(f"\nTrying mutation rate: {current_mutation_rate}")

        # Run with new mutation rate (fewer generations for efficiency)
        new_best_schedule, _, _ = genetic_algorithm(population_size, 200, current_mutation_rate)
        new_best_fitness = new_best_schedule.fitness

        print(f"Best fitness: {new_best_fitness:.2f}")

        # Check if improvement is significant (>= 1%)
        improvement = (new_best_fitness - previous_best_fitness) / abs(previous_best_fitness)
        improved = improvement >= 0.01

        if improved:
            print(f"Improvement: {improvement:.2%}, continuing with lower mutation rate")
            previous_best_fitness = new_best_fitness
            best_schedule = new_best_schedule  # Update best schedule
            current_mutation_rate /= 2  # Halve mutation rate
        else:
            print(f"Improvement: {improvement:.2%} < 1%, stopping")

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"\nFinal best fitness: {best_schedule.fitness:.2f}")
    print(f"Final mutation rate: {current_mutation_rate * 2:.6f}")  # The last one that showed improvement
    print(f"Execution time: {execution_time:.2f} seconds")

    # Print schedule to file
    print_schedule(best_schedule, "final_schedule.txt")

if __name__ == "__main__":
    main()