import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_annotations", type=str, help="Path to the train annotations npy file.")
    parser.add_argument("train_texts", type=str, help="Path to the train texts npy file.")
    parser.add_argument("val_annotations", type=str, help="Path to the validation annotations npy file.")
    parser.add_argument("val_texts", type=str, help="Path to the validation texts npy file.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train for.")
    parser.add_argument("--encoder_model", type=str, default="bert-base-cased", help="HuggingFace model to use.")
    parser.add_argument("--model_type", type=str, choices=["multi", "base"], help="Type of predictor to use.")
    parser.add_argument("--logger_type", type=str, choices=["wandb", "tensorboard"], default="tensorboard", help="Type of logger to use.")
    parser.add_argument("--wandb_project", help='wandb project name', default=None, type=str)
    parser.add_argument("--wandb_run_name", help='wandb run name', default='default_run_name', type=str)
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--save_logs", action="store_true", help="Whether to save the json logs.")
    parser.add_argument("--save_logs_where", type=str, default="logs", help="Path to save json logs. Only used if save_logs is True.")
    
    args = parser.parse_args()
    return args