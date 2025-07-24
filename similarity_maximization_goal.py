from textattack.goal_functions.classification import ClassificationGoalFunction
from textattack.shared import AttackedText

class SimilarityMaximizationGoal(ClassificationGoalFunction):
    def __init__(self, model, threshold=0.67):
        super().__init__(model)
        self.threshold = threshold

    def _is_goal_complete(self, model_output, attacked_text):
        # model_output: [match_prob, non_match_prob]
        match_prob = model_output[0]
        return match_prob >= self.threshold

    def _get_score(self, model_output, attacked_text):
        return model_output[0]