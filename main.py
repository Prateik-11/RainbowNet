"""
Main function used to train the Cassava Leaf Disease Detection model. 
Utilizes file constants.py as well as the following hyperparameters: 

- Epochs - how many times should the model pass through the dataset? 
- Batch size - how many images to use in one training iteration? 
- n_eval - how often should we evaluate the model (in iterations)? 
- logdir - where should we output our tensorboard files to? 
- datadir - where is the cassava leaf disease dataset (6 GB)? 
- summaries - where should we output our training summaries to? 

"""

import os, torch

from StartingDataset import StartingDataset 
from networks.StartingNetwork import StartingNetwork
from starting_train import starting_train


def main():
    # Get command line arguments
    # args = parse_arguments()
    # hyperparameters = {"epochs": args.epochs, "batch_size": args.batch_size, "data_dir": args.datadir}

    # # Create path for training summaries
    # summary_path = None
    # if args.logdir is not None:
    #     summary_path = f"{args.summaries}/{args.logdir}"
    #     os.makedirs(summary_path, exist_ok=True)

    # Use GPU, if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # Print out training information 
    # print("\n----------------- Summary Information --------------------")
    # print(f"\nSummary path: {summary_path}")
    # print(f"   Data path: {constants.DATA_DIR}\n")
    # print(f"Epochs: {args.epochs}")
    # print(f"    Evaluate: {constants.N_EVAL}")
    # print(f"  Save Model: {constants.SAVE_INTERVAL} \n")

    # print(f"Training imgs: {constants.TRAIN_NUM * constants.IMG_TYPES}")
    # print(f" Testing imgs: {constants.TEST_NUM * constants.IMG_TYPES}")
    # print(f"   Batch size: {args.batch_size}\n")
    # print("----------------------------------------------------------")


    # # Initalize dataset and model. Then train the model!
    # csv_path = args.datadir +'/train.csv'
    train_dataset = StartingDataset()
    val_dataset = StartingDataset(training_set = False)

    # Create our model, and begin starting_train()
    model = StartingNetwork()
    model = torch.nn.DataParallel(model)
    starting_train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model=model,
    )


if __name__ == "__main__":
    main()
