import os
import time
import argparse
import numpy as np
import traceback

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from sklearn.metrics import mean_squared_error

import model_ours
import config
import evaluate
import data_utils_ours
from utils import Logger

try:
    args = config.parse_args()

    logger = Logger(f'{args.output_path.split("_")[0]}_{args.fold}')
    logger.log(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cudnn.benchmark = True

    ############################## PREPARE DATASET ##########################
    (
        train_data,
        test_data,
        item_num,
        user_num,
        train_mat,
        train_labels,
        test_labels,
        infer_data,
        infer_labels,
    ) = data_utils_ours.load_all_ghc(args)

    # construct the train and test datasets
    train_dataset = data_utils_ours.NCFData(
        train_data, train_labels, item_num, train_mat, args.num_ng, True
    )
    test_dataset = data_utils_ours.NCFData(
        test_data, test_labels, item_num, train_mat, 0, False
    )
    infer_dataset = data_utils_ours.NCFData(
        infer_data, infer_labels, item_num, train_mat, 0, False
    )

    train_loader = data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    test_loader = data.DataLoader(
        test_dataset, batch_size=args.test_num_ng + 1, shuffle=False, num_workers=0
    )
    infer_loader = data.DataLoader(
        infer_dataset, batch_size=args.test_num_ng + 1, shuffle=False, num_workers=0
    )

    ########################### CREATE MODEL #################################
    if config.model == "NeuMF-pre":
        assert os.path.exists(config.GMF_model_path), "lack of GMF model"
        assert os.path.exists(config.MLP_model_path), "lack of MLP model"
        GMF_model = torch.load(config.GMF_model_path)
        MLP_model = torch.load(config.MLP_model_path)
    else:
        GMF_model = None
        MLP_model = None

    ########################### Hyper-parameter search #################################

    factors = [4, 8, 16, 32, 64, 128]
    lrs = [0.001, 0.0005, 0.0001, 0.00005]
    best_rmse = 10.0
    best_factor = -1
    best_lr = -1
    #dataset_name = data_utils_ours.get_dataset_name_from_input_path(args.input_path)
    max_label = max(np.unique(train_labels))

    for factor in factors:
        for lr in lrs:
            logger.log(f"Factor: {factor}, LR: {lr}")
            args.factor_num = factor
            args.lr = lr

            model = model_ours.NCF(
                user_num,
                item_num,
                args.factor_num,
                args.num_layers,
                args.dropout,
                config.model,
                GMF_model,
                MLP_model,
            )
            model.cuda()
            loss_function = nn.MSELoss()

            if config.model == "NeuMF-pre":
                optimizer = optim.SGD(model.parameters(), lr=args.lr)
            else:
                optimizer = optim.Adam(model.parameters(), lr=args.lr)

            # writer = SummaryWriter() # for visualization

            ########################### TRAINING #####################################
            count = 0
            model.eval()
            rmse = evaluate.rmse(model, test_loader, max_label)

            # print("RMSE: {:.3f}".format(rmse))

            for epoch in range(args.epochs):
                model.train()  # Enable dropout (if have).
                start_time = time.time()

                all_labels = []
                all_preds = []

                for user, item, label in train_loader:
                    user = user.cuda()
                    item = item.cuda()
                    label = label.float().cuda()

                    model.zero_grad()
                    prediction = model(user, item)
                    loss = loss_function(prediction, label)
                    loss.backward()
                    optimizer.step()
                    # writer.add_scalar('data/loss', loss.item(), count)
                    count += 1

                    all_labels.append(label.detach().cpu())
                    all_preds.append(prediction.detach().cpu())
                all_labels = torch.cat(all_labels, dim=0).numpy()
                all_preds = torch.cat(all_preds, dim=0).numpy()
                rmse_train = mean_squared_error(all_labels, all_preds, squared=False)
                # print("RMSE (train): {:.3f}".format(rmse_train))

                model.eval()
                rmse = evaluate.rmse(model, test_loader, max_label)

                elapsed_time = time.time() - start_time
                print(
                    "The time elapse of epoch {:03d}".format(epoch)
                    + " is: "
                    + str(elapsed_time)
                )

                if rmse < best_rmse:
                    print("RMSE (test): {:.3f}".format(rmse))
                    logger.log("RMSE (test): {:.3f}".format(rmse))
                    best_rmse = rmse
                    best_factor = factor
                    best_lr = lr
                    imputed_anno = evaluate.get_imputed_outputs(
                        model, train_mat, infer_loader, max_label 
                    )
                    np.save(args.output_path + "_" + str(args.fold) + ".npy", imputed_anno)

    print("End. Best RMSE = {:.3f}".format(best_rmse))
    logger.log("End. Best RMSE = {:.3f}".format(best_rmse))
    logger.log(f"Best Factor: {best_factor}, Best LR: {best_lr}")
except Exception as e:
    print("Error encountered!")
    print(e)
    # print the traceback of the error
    print(traceback.format_exc())
    raise