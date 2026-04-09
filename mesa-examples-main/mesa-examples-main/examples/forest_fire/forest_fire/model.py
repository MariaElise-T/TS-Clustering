import mesa
from .agent import TreeCell


class ForestFire(mesa.Model):
    """
    Simple Forest Fire model.
    """

    def __init__(self, width=100, height=100, density=0.65):
        super().__init__()

        self.width = width
        self.height = height

        # Scheduler (works in Mesa 1.x)
        self.schedule = mesa.time.RandomActivation(self)

        # Grid
        self.grid = mesa.space.SingleGrid(width, height, torus=False)

        # Data collector
        self.datacollector = mesa.DataCollector(
            {
                "Fine": lambda m: self.count_type(m, "Fine"),
                "On Fire": lambda m: self.count_type(m, "On Fire"),
                "Burned Out": lambda m: self.count_type(m, "Burned Out"),
            }
        )

        # ---------------------------
        # Populate grid with trees
        # ---------------------------
        for x in range(width):
            for y in range(height):

                if self.random.random() < density:

                    # IMPORTANT: correct Mesa-style agent creation
                    new_tree = TreeCell((x, y), self)

                    # ignite left edge
                    if x == 0:
                        new_tree.condition = "On Fire"

                    self.grid.place_agent(new_tree, (x, y))
                    self.schedule.add(new_tree)

        self.running = True
        self.datacollector.collect(self)

    def step(self):
        """
        Advance the model by one step.
        """
        self.schedule.step()
        self.datacollector.collect(self)

        # Stop if no fire remains
        if self.count_type(self, "On Fire") == 0:
            self.running = False

    @staticmethod
    def count_type(model, tree_condition):
        """
        Count trees in a given state.
        """
        return sum(
            1 for agent in model.schedule.agents
            if agent.condition == tree_condition
        )