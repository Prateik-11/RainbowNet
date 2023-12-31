"""
Helper functions used to train the model: 
- starting_train: training loop used to train the model 
- compute_accuracy: returns difference between images and labels 
- evaluate: run the model with the validation dataset

"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensorboard

global device
device = 'cuda' 

import numpy as np


"""
Trains and evaluates a model.

Args:
    train_dataset:   PyTorch dataset containing training data.
    val_dataset:     PyTorch dataset containing validation data.
    model:           PyTorch model to be trained.
    hyperparameters: Dictionary containing hyperparameters.
    n_eval:          Interval at which we evaluate our model.
    summary_path:    Path where Tensorboard summaries are located.
"""

def starting_train(
    train_dataset, val_dataset, model
):

    batch_size = 512
    epochs = 50
    n_eval = 10

    # Initialize dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )

    # Switch to Cuda (GPU), if available
    # if torch.cuda.is_available():
    #     device = torch.device('cuda:0')
    # else:
    #     device = torch.device('cpu')

    model = model.to(device)

    # Initalize optimizer (for gradient descent) and loss function
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.L1Loss() # TODO change me

    ckpt = torch.load('/localhome/advit/rainbow_train/leaf-us-alone/checkpoints/model.pt')
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    print('Loaded checkpoint for model and optimizer')

    # Initialize summary writer (for logging)
    # tb_summary = None
    # if summary_path is not None:
    #     tb_summary = torch.utils.tensorboard.SummaryWriter(summary_path)


    # Begin training loop (number of epochs)
    step = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")

        # Loop over each batch in the dataset
        for i, batch in enumerate(train_loader):
            print(f"\rIteration {i + 1} of {len(train_loader)} ...", end="")

            input_data, label_data = batch
            input_data = input_data.to(device)
            label_data = label_data.to(device)
            
            pred = model(input_data)

            pred = pred.squeeze()

            #print(pred.shape)
            #print(label_data.shape)
            # Prediction, label data have same shape
            loss = loss_fn(pred, label_data) 
            #pred = pred.argmax(axis=1)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print(f"\n    Train Loss: {loss.item()}")

            # Periodically evaluate our model + log to Tensorboard
            if (step + 1) % n_eval == 0:
                # Compute training loss and accuracy.
                train_accuracy = compute_accuracy(pred, label_data)
                print(f"    Train Accu: {train_accuracy}")

                # # Log the results to Tensorboard
                # if tb_summary:
                #     tb_summary.add_scalar('Loss (Training)', loss, epoch)
                #     tb_summary.add_scalar('Accuracy (Training)', train_accuracy, epoch)

                # Compute validation loss and accuracy.
                valid_loss, valid_accuracy = evaluate(val_loader, model, loss_fn)

                # # Log the results to Tensorboard.
                # if tb_summary:
                #     tb_summary.add_scalar('Loss (Validation)', valid_loss, epoch)
                #     tb_summary.add_scalar('Accuracy (Validation)', valid_accuracy, epoch)

                print(f"    Valid Loss: {valid_loss}")
                print(f"    Valid Accu: {valid_accuracy}")

                model.train()

                save_path = '/localhome/advit/rainbow_train/leaf-us-alone/checkpoints/model.pt'
                torch.save({
                    'model' : model.state_dict(),
                    'optimizer' : optimizer.state_dict()
                }, save_path)

            step += 1

        print()

        # # If we have a save interval and we are past it, save the model
        # if constants.SAVE_INTERVAL and (epoch + 1) % constants.SAVE_INTERVAL:
        #     print("Saving model...")
        #     torch.save(model.state_dict(), constants.SAVE_DIR)

        # print()


"""
Computes the accuracy of a model's predictions.

Example input:
    outputs: [0.7, 0.9, 0.3, 0.2]
    labels:  [1, 1, 0, 1]

Example output:
    0.75
"""

def compute_accuracy(outputs, labels):
    #print(outputs.shape)
    #print(labels.shape)
    #print("Computing accuracy")
    #print(type((outputs - labels)**2))
    #print((outputs - labels)**2)
    with torch.no_grad():
        softmax = torch.nn.Softmax2d()
        outputs = torch.round(softmax(outputs))
        labels = torch.round(softmax(labels))

        return torch.mean((outputs == labels).float())


"""
Computes the loss and accuracy of a model on the validation dataset.

"""

def evaluate(val_loader, model, loss_fn):

    model.eval()
    model = model.to(device)
    #model.resnet = (model.resnet).to(device)

    loss, acc = 0, [] 

    with torch.no_grad(): 
        for batch in val_loader:
            input_data, label_data = batch

            # Move both images and labels to GPU, if available
            input_data = input_data.to('cuda')
            label_data = label_data.to('cuda')

            pred = model(input_data)
            pred = pred.squeeze() 

            loss += loss_fn(pred, label_data).mean().item()

            acc.append(compute_accuracy(pred, label_data))


    acc = acc[0].item()
    return loss, acc
