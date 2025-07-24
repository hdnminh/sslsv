import os
import sys

from dataclasses import dataclass, field

# Set up paths so we can import notebooks_utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'notebooks'))

from notebooks.notebooks_utils import load_models, evaluate_models, create_metrics_df
from sslsv.evaluations.CosineSVEvaluation import CosineSVEvaluation, CosineSVEvaluationTaskConfig

if __name__ == "__main__":
    # Paths to config and checkpoint
    config = "models/tests/simclr/simclr_e-ecapa/ssps_kmeans_25k_uni-1/config.yml"
    checkpoint = "models/tests/simclr/simclr_e-ecapa/ssps_kmeans_25k_uni-1/checkpoints/model_avg.pt"

    # Load model from config and checkpoint
    models = load_models([config], checkpoint_name=os.path.basename(checkpoint))

    # Patch model configs to avoid memory issues
    # for model_entry in models.values():
    #     model_entry.config.dataset.num_workers = 0
    #     model_entry.config.dataset.pin_memory = False

    # Patch evaluation config to further reduce memory usage
    eval_task_config = CosineSVEvaluationTaskConfig(
        __type__="sv_cosine",
        frame_length=None,  # 0.5s at 16kHz
        num_frames=1,       # Only one frame per file
    )

    # CosineSVEvaluationTaskConfig(
    # __type__='sv_cosine', 
    #  __subtype__=None, 
    # batch_size=64, 
    # num_frames=7, 
    # frame_length=64000, 
    # trials=['voxceleb1_test_O'], 
    # metrics=['eer', 'mindcf'], 
    # mindcf_p_target=0.01, 
    # mindcf_c_miss=1, 
    # mindcf_c_fa=1, 
    # score_norm=<ScoreNormEnum.NONE: None>, score_norm_cohort_size=20000)

    # Evaluate (type: ignore for linter)
    evaluate_models(models, CosineSVEvaluation, eval_task_config)  # type: ignore

    # Debug: print metric keys for each model  
    for model_name, model_entry in models.items():
        print(f"Model: {model_name}, metric keys: {list(getattr(model_entry, 'metrics', {}).keys())}")

    # Print metrics
    df = create_metrics_df(models)
    print(df)




