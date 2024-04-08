import sys
import time
from imblearn.over_sampling import SMOTE
import os
import pandas as pd
from torch.autograd import Variable
from sklearn import metrics
from sklearn.model_selection import KFold
import random
os.chdir(sys.path[0])
from DeepHotResi import *
# import dgl
import torch.nn as nn

# Path
Dataset_Path = "./Dataset/"
Model_Path = "./Model/"
Log_path = "./Log/"
model_time = None


def train_one_epoch(model, data_loader):
    epoch_loss_train = 0.0
    n = 0
    for i,data in enumerate(data_loader):
        model.optimizer.zero_grad()
        _, _, labels, node_features, G_batch, adj_matrix = data

        if torch.cuda.is_available():
            node_features = Variable(node_features.cuda().float())
            G_batch.edata['ex'] = Variable(G_batch.edata['ex'].float())
            G_batch = G_batch.to(torch.device('cuda:0'))
            adj_matrix = Variable(adj_matrix.cuda())
            y_true = Variable(labels.cuda())
        else:
            node_features = Variable(node_features.float())
            G_batch.edata['ex'] = Variable(G_batch.edata['ex'].float())
            adj_matrix = Variable(adj_matrix)
            y_true = Variable(labels)

        adj_matrix = torch.squeeze(adj_matrix)
        y_true = torch.squeeze(y_true)
        y_true = y_true.long()

        y_pred = model(node_features, G_batch, adj_matrix)

        # 平衡y标签
        y_true_positive_index = torch.nonzero(y_true == 1)
        negitve_index = torch.nonzero(y_true == 0)
        negitve_index = random.sample(negitve_index.tolist(), len(y_true_positive_index))
        train_index = y_true_positive_index.flatten().tolist()+negitve_index[0]

        # calculate loss
        loss = model.criterion(y_pred[train_index], y_true[train_index])
        if i % 4==0:

            # backward gradient
            loss.backward()

            # update all parameters
            model.optimizer.step()

        epoch_loss_train += loss.item()
        n += 1

    epoch_loss_train_avg = epoch_loss_train / n
    return epoch_loss_train_avg


def evaluate(model, data_loader):
    model.eval()
    epoch_loss = 0.0
    n = 0
    valid_pred = []
    valid_true = []
    pred_dict = {}

    for data in data_loader:
        with torch.no_grad():
            sequence_names, _, labels, node_features, G_batch, adj_matrix = data

            if torch.cuda.is_available():
                node_features = Variable(node_features.cuda().float())
                adj_matrix = Variable(adj_matrix.cuda())
                G_batch.edata['ex'] = Variable(G_batch.edata['ex'].float())
                G_batch = G_batch.to(torch.device('cuda:0'))
                y_true = Variable(labels.cuda())

            else:
                node_features = Variable(node_features.float())
                adj_matrix = Variable(adj_matrix)
                y_true = Variable(labels)
                G_batch.edata['ex'] = Variable(G_batch.edata['ex'].float())

            adj_matrix = torch.squeeze(adj_matrix)
            y_true = torch.squeeze(y_true)
            y_true = y_true.long()

            y_pred = model(node_features, G_batch, adj_matrix)

            # 平衡y标签
            y_true_positive_index = torch.nonzero(y_true == 1)
            negitve_index = torch.nonzero(y_true == 0)
            negitve_index = random.sample(negitve_index.tolist(), len(y_true_positive_index))
            negitve_index = [index for sublist in negitve_index for index in sublist]
            valid_index = y_true_positive_index.flatten().tolist()+negitve_index
                                                                                


            # calculate loss
            y_pred, y_true  = y_pred[valid_index], y_true[valid_index]
            loss = model.criterion(y_pred, y_true)
            softmax = torch.nn.Softmax(dim=1)
            y_pred = softmax(y_pred)
            y_pred = y_pred.cpu().detach().numpy()
            y_true = y_true.cpu().detach().numpy()
            valid_pred += [pred[1] for pred in y_pred]
            valid_true += list(y_true)
            pred_dict[sequence_names[0]] = [pred[1] for pred in y_pred]

            epoch_loss += loss.item()
            n += 1
    epoch_loss_avg = epoch_loss / n

    

    return epoch_loss_avg, valid_true, valid_pred, pred_dict


def analysis(y_true, y_pred, best_threshold = None):
    
    
    if best_threshold is None:
        best_auc = 0
        best_threshold = 0
        for threshold in range(0, 1001):  # 步长改为0.001
            threshold = threshold / 1000
            binary_pred = [1 if pred >= threshold else 0 for pred in y_pred]
            binary_true = y_true
            auc = metrics.roc_auc_score(binary_true, y_pred)
            if auc > best_auc:
                best_auc = auc
                best_threshold = threshold

    binary_pred = [1 if pred >= best_threshold else 0 for pred in y_pred]
    binary_true = y_true

    # binary evaluate
    binary_acc = metrics.accuracy_score(binary_true, binary_pred)
    precision = metrics.precision_score(binary_true, binary_pred)
    recall = metrics.recall_score(binary_true, binary_pred)
    f1 = metrics.f1_score(binary_true, binary_pred)
    AUC = metrics.roc_auc_score(binary_true, y_pred)
    precisions, recalls, thresholds = metrics.precision_recall_curve(binary_true, y_pred)
    AUPRC = metrics.auc(recalls, precisions)
    mcc = metrics.matthews_corrcoef(binary_true, binary_pred)

    results = {
        'binary_acc': binary_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'AUC': AUC,
        'AUPRC': AUPRC,
        'mcc': mcc,
        'threshold': best_threshold
    }
    return results


def train(model, train_dataframe, valid_dataframe, fold = 0):
    train_loader = DataLoader(dataset=ProDataset(train_dataframe), batch_size=BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=graph_collate)
    valid_loader = DataLoader(dataset=ProDataset(valid_dataframe), batch_size=BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=graph_collate)

    best_epoch = 0
    best_val_auc = 0
    best_val_aupr = 0

    for epoch in range(NUMBER_EPOCHS):
        print("\n========== Train epoch " + str(epoch + 1) + " ==========")
        model.train()

        epoch_loss_train_avg = train_one_epoch(model, train_loader)
        print("========== Evaluate Train set ==========")
        _, train_true, train_pred, _ = evaluate(model, train_loader)
        result_train = analysis(train_true, train_pred, 0.5)
        print("Train loss: ", epoch_loss_train_avg)
        print("Train binary acc: ", result_train['binary_acc'])
        print("Train AUC: ", result_train['AUC'])
        print("Train AUPRC: ", result_train['AUPRC'])

        print("========== Evaluate Valid set ==========")
        epoch_loss_valid_avg, valid_true, valid_pred, _ = evaluate(model, valid_loader)
        result_valid = analysis(valid_true, valid_pred, 0.5)
        print("Valid loss: ", epoch_loss_valid_avg)
        print("Valid binary acc: ", result_valid['binary_acc'])
        print("Valid precision: ", result_valid['precision'])
        print("Valid recall: ", result_valid['recall'])
        print("Valid f1: ", result_valid['f1'])
        print("Valid AUC: ", result_valid['AUC'])
        print("Valid AUPRC: ", result_valid['AUPRC'])
        print("Valid mcc: ", result_valid['mcc'])

        if best_val_aupr < result_valid['AUPRC']:
            best_epoch = epoch + 1
            best_val_auc = result_valid['AUC']
            best_val_aupr = result_valid['AUPRC']
            torch.save(model.state_dict(), os.path.join(Model_Path, 'Fold' + str(fold) + '_best_model.pkl'))

        model.scheduler.step(result_valid['AUPRC'])

    return best_epoch, best_val_auc, best_val_aupr


def cross_validation(all_dataframe, fold_number=5):
    print("Random seed:", SEED)
    print("Map cutoff:", MAP_CUTOFF)
    print("Feature dim:", INPUT_DIM)
    print("Hidden dim:", HIDDEN_DIM)
    print("Dropout:", DROPOUT)
    print("Learning rate:", LEARNING_RATE)
    print("Training epochs:", NUMBER_EPOCHS)
    print()

 
    sequence_names = all_dataframe['ID'].values
    sequence_labels = all_dataframe['label'].values
    kfold = KFold(n_splits=fold_number, shuffle=True)
    fold = 0
    best_epochs = []
    valid_aucs = []
    valid_auprs = []

    for train_index, valid_index in kfold.split(sequence_names, sequence_labels):
        print("\n\n========== Fold " + str(fold + 1) + " ==========")
        train_dataframe = all_dataframe.iloc[train_index, :]
        valid_dataframe = all_dataframe.iloc[valid_index, :]
        print("Train on", str(train_dataframe.shape[0]), "samples, validate on", str(valid_dataframe.shape[0]),
              "samples")

        model = DeepHotResi(INPUT_DIM, HIDDEN_DIM, NUM_CLASSES, DROPOUT)
        if torch.cuda.is_available():
            model.cuda()

        best_epoch, valid_auc, valid_aupr = train(model, train_dataframe, valid_dataframe, fold + 1)
        best_epochs.append(str(best_epoch))
        valid_aucs.append(valid_auc)
        valid_auprs.append(valid_aupr)
        fold += 1

    print("\n\nBest epoch: " + " ".join(best_epochs))
    print("Average AUC of {} fold: {:.4f}".format(fold_number, sum(valid_aucs) / fold_number))
    print("Average AUPR of {} fold: {:.4f}".format(fold_number, sum(valid_auprs) / fold_number))
    return round(sum([int(epoch) for epoch in best_epochs]) / fold_number)


def train_full_model(all_dataframe, aver_epoch):
    print("\n\nTraining a full model using all training data...\n")
    model = DeepHotResi(INPUT_DIM, HIDDEN_DIM, NUM_CLASSES, DROPOUT)
    if torch.cuda.is_available():
        model.cuda()

    train_loader = DataLoader(dataset=ProDataset(all_dataframe), batch_size=BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=graph_collate)

    for epoch in range(NUMBER_EPOCHS):
        print("\n========== Train epoch " + str(epoch + 1) + " ==========")
        model.train()

        epoch_loss_train_avg = train_one_epoch(model, train_loader)
        print("========== Evaluate Train set ==========")
        _, train_true, train_pred, _ = evaluate(model, train_loader)
        result_train = analysis(train_true, train_pred, 0.5)
        print("Train loss: ", epoch_loss_train_avg)
        print("Train binary acc: ", result_train['binary_acc'])
        print("Train AUC: ", result_train['AUC'])
        print("Train AUPRC: ", result_train['AUPRC'])

        if epoch + 1 in [aver_epoch, 45]:
            torch.save(model.state_dict(), os.path.join(Model_Path, 'Full_model_{}.pkl'.format(epoch + 1)))  # 保存模型的参数


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, 'ab', buffering=0)

    def write(self, message):
        self.terminal.write(message)
        try:
            self.log.write(message.encode('utf-8'))
        except ValueError:
            pass

    def close(self):
        self.log.close()
        sys.stdout = self.terminal

    def flush(self):
        pass


def main():
    if not os.path.exists(Log_path): os.makedirs(Log_path)

    with open(Dataset_Path + "data_dict.pkl", "rb") as f:
        Train_set = pickle.load(f)
        Train_set.pop('1N78_A')  

        Train_set.pop('5HO4_A') 
        Train_set.pop('1WNE_A')  
        Train_set.pop('4JVH_A')  
        Train_set.pop('2XB2_G')  
        Train_set.pop('4G0A_A')  
        Train_set.pop('1ZDI_A')  
        Train_set.pop('4CIO_A')  

        Train_set.pop('1C9S_L')   
        Train_set.pop('2KXN_B') 
        Train_set.pop('2XS2_A')   
        Train_set.pop('5DNO_A')    
        Train_set.pop('1T0K_A')    
    
    IDs, sequences, labels = [], [], []

    for ID in Train_set:
        IDs.append(ID)
        item = Train_set[ID]
        sequences.append(item[0])
        labels.append(item[1])

    train_dic = {"ID": IDs, "sequence": sequences, "label": labels}
    train_dataframe = pd.DataFrame(train_dic)
    aver_epoch = cross_validation(train_dataframe, fold_number=5)
    train_full_model(train_dataframe, aver_epoch)

if __name__ == "__main__":

    if model_time is not None:
        checkpoint_path = os.path.normpath(Log_path +"/"+ model_time)
    else:
        localtime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        checkpoint_path = os.path.normpath(Log_path + "/" + localtime)
        os.makedirs(checkpoint_path)
    Model_Path = os.path.normpath(checkpoint_path + '/model')
    if not os.path.exists(Model_Path): os.makedirs(Model_Path)

    sys.stdout = Logger(os.path.normpath(checkpoint_path + '/training.log'))
    main()
    sys.stdout.log.close()
